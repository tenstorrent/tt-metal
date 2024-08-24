// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"
#include "ttnn/operations/core/work_split/work_split.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/matmul/device/matmul_op.hpp"

using namespace tt::constants;
using namespace tt;
using tt_metal::Buffer;

tt_metal::operation::ProgramWithCallbacks create_program(
    tt_metal::Device *device,
    tt::DataFormat in0_cb_data_format,
    tt::DataFormat in1_cb_data_format,
    tt::DataFormat out_cb_data_format,
    MathFidelity math_fidelity,
    uint32_t num_cores_x,
    uint32_t B,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    bool bcast_batch,
    uint32_t in0_block_w,
    uint32_t out_subblock_h,
    uint32_t out_subblock_w,
    uint32_t per_core_M,
    uint32_t per_core_N,
    tt_metal::Buffer *in0_buffer,
    tt_metal::Buffer *in1_buffer,
    tt_metal::Buffer *out_buffer) {
    tt_metal::Program program{};

    uint32_t in0_single_tile_size = tt_metal::detail::TileSize(in0_cb_data_format);
    uint32_t in1_single_tile_size = tt_metal::detail::TileSize(in1_cb_data_format);
    uint32_t out_single_tile_size = tt_metal::detail::TileSize(out_cb_data_format);

    uint32_t in0_block_tiles = per_core_M * in0_block_w;
    uint32_t in0_CB_tiles = in0_block_tiles * 2;  // double buffer
    uint32_t in0_CB_size = in0_CB_tiles * in0_single_tile_size;
    uint32_t in1_block_tiles = per_core_N * in0_block_w;
    uint32_t in1_CB_tiles = in1_block_tiles * 2;  // double buffer
    uint32_t in1_CB_size = in1_CB_tiles * in1_single_tile_size;
    uint32_t out_block_tiles = per_core_M * per_core_N;
    uint32_t out_CB_tiles = out_block_tiles;  // No double buffer
    uint32_t out_CB_size = out_CB_tiles * out_single_tile_size;

    // Compute kernel compile time args
    uint32_t num_blocks = (K / in0_block_w);

    uint32_t in0_num_subblocks = (per_core_M / out_subblock_h);
    uint32_t in0_block_num_tiles = out_subblock_h * in0_block_w * in0_num_subblocks;
    uint32_t in0_subblock_num_tiles = out_subblock_h * in0_block_w;

    uint32_t in1_num_subblocks = (per_core_N / out_subblock_w);
    uint32_t in1_block_num_tiles = out_subblock_w * in0_block_w * in1_num_subblocks;
    uint32_t in1_per_core_w = out_subblock_w * in1_num_subblocks;

    uint32_t out_subblock_num_tiles = out_subblock_h * out_subblock_w;

    vector<uint32_t> compute_kernel_args = {
        in0_block_w,             // in0_block_w
        in0_num_subblocks,       // in0_num_subblocks
        in0_block_num_tiles,     // in0_block_num_tiles
        in0_subblock_num_tiles,  // in0_subblock_num_tiles

        in1_num_subblocks,    // in1_num_subblocks
        in1_block_num_tiles,  // in1_block_num_tiles
        in1_per_core_w,       // in1_per_core_w

        num_blocks,  // num_blocks

        out_subblock_h,          // out_subblock_h
        out_subblock_w,          // out_subblock_w
        out_subblock_num_tiles,  // out_subblock_num_tiles
        B                        // batch
    };

    uint32_t num_blocks_read = 0;
    uint32_t num_blocks_y = M / per_core_M;
    uint32_t num_blocks_x = N / per_core_N;

    CoreRangeSet all_cores(ttnn::num_cores_to_corerange_set(
        num_blocks_x * num_blocks_y, device->compute_with_storage_grid_size(), true));

    // Create circular buffers
    uint32_t src0_cb_index = 0;
    tt_metal::CircularBufferConfig src0_cb_config =
        tt_metal::CircularBufferConfig(in0_CB_size, {{src0_cb_index, in0_cb_data_format}})
            .set_page_size(src0_cb_index, in0_single_tile_size);
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, all_cores, src0_cb_config);

    uint32_t src1_cb_index = 1;
    tt_metal::CircularBufferConfig src1_cb_config =
        tt_metal::CircularBufferConfig(in1_CB_size, {{src1_cb_index, in1_cb_data_format}})
            .set_page_size(src1_cb_index, in1_single_tile_size);
    auto cb_src1 = tt_metal::CreateCircularBuffer(program, all_cores, src1_cb_config);

    uint32_t output_cb_index = 16;  // output operands start at index 16
    uint32_t interm0_cb_index = 24;
    std::map<uint8_t, tt::DataFormat> output_cb_data_format_spec{
        {output_cb_index, out_cb_data_format}, {interm0_cb_index, out_cb_data_format}};
    tt_metal::CircularBufferConfig output_cb_config =
        tt_metal::CircularBufferConfig(out_CB_size, output_cb_data_format_spec)
            .set_page_size(output_cb_index, out_single_tile_size)
            .set_page_size(interm0_cb_index, out_single_tile_size);
    auto cb_output = tt_metal::CreateCircularBuffer(program, all_cores, output_cb_config);

    bool in0_is_dram = in0_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool in1_is_dram = in1_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> reader_compile_time_args = {(uint32_t)in0_is_dram, (uint32_t)in1_is_dram};

    bool out_is_dram = out_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> writer_compile_time_args = {(uint32_t)out_is_dram};

    // Create reader and writer kernels per core
    auto mm_reader_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout.cpp",
        all_cores,
        tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    auto unary_writer_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/writer_bmm_tile_layout.cpp",
        all_cores,
        tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    // Create compute kernel
    auto mm_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm.cpp",
        all_cores,
        tt_metal::ComputeConfig{.math_fidelity = math_fidelity, .compile_args = compute_kernel_args});

    for (int output_idx_y = 0; output_idx_y < num_blocks_y; output_idx_y++) {
        for (int output_idx_x = 0; output_idx_x < num_blocks_x; output_idx_x++) {
            int core_idx_x = num_blocks_read % num_cores_x;
            int core_idx_y = num_blocks_read / num_cores_x;
            CoreCoord core = {(std::size_t)core_idx_x, (std::size_t)core_idx_y};

            // Write runtime args to device
            std::vector<uint32_t> mm_reader_args = {
                (std::uint32_t)in0_buffer->address(),          // in0_tensor_addr
                (std::uint32_t)K * per_core_M * output_idx_y,  // in0_tensor_start_tile_id
                (std::uint32_t)1,                              // in0_tensor_stride_w
                (std::uint32_t)K,                              // in0_tensor_stride_h
                (std::uint32_t)in0_block_w,                    // in0_tensor_next_block_stride

                (std::uint32_t)in0_block_w,               // in0_block_w
                (std::uint32_t)per_core_M,                // in0_block_h
                (std::uint32_t)in0_block_w * per_core_M,  // in0_block_num_tiles

                (std::uint32_t)in1_buffer->address(),      // in1_tensor_addr
                (std::uint32_t)per_core_N * output_idx_x,  // in1_tensor_start_tile_id
                (std::uint32_t)1,                          // in1_tensor_stride_w
                (std::uint32_t)N,                          // in1_tensor_stride_h
                (std::uint32_t)in0_block_w * N,            // in1_tensor_next_block_stride

                (std::uint32_t)per_core_N,                // in1_block_w
                (std::uint32_t)in0_block_w,               // in1_block_h
                (std::uint32_t)per_core_N * in0_block_w,  // in1_block_num_tiles

                (std::uint32_t)K / in0_block_w,  // num_blocks

                (std::uint32_t)M * K,       // MtKt
                (std::uint32_t)K * N,       // KtNt
                (std::uint32_t)B,           // batch
                (std::uint32_t)bcast_batch  // bcast_B
            };

            std::vector<uint32_t> writer_args = {
                (std::uint32_t)out_buffer->address(),                                      // out_tensor_addr
                (std::uint32_t)output_idx_x * per_core_N + output_idx_y * per_core_M * N,  // out_tensor_start_tile_id
                (std::uint32_t)1,                                                          // out_tensor_stride_w
                (std::uint32_t)N,                                                          // out_tensor_stride_h
                (std::uint32_t)out_subblock_w,      // out_tensor_next_subblock_stride_w
                (std::uint32_t)out_subblock_h * N,  // out_tensor_next_subblock_stride_h

                (std::uint32_t)out_subblock_w,                     // out_subblock_w
                (std::uint32_t)out_subblock_h,                     // out_subblock_h
                (std::uint32_t)(out_subblock_w * out_subblock_h),  // out_subblocks_w * out_subblocks_h
                (std::uint32_t)(per_core_N / out_subblock_w),      // out_num_subblocks_w
                (std::uint32_t)(per_core_M / out_subblock_h),      // out_num_subblocks_h

                (std::uint32_t)M * N,  // MtNt
                (std::uint32_t)B       // batch
            };

            tt_metal::SetRuntimeArgs(program, mm_reader_kernel_id, core, mm_reader_args);
            tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, writer_args);

            num_blocks_read++;
        }
    }

    auto override_runtime_args_callback = [reader_kernel_id = mm_reader_kernel_id,
                                           writer_kernel_id = unary_writer_kernel_id,
                                           num_cores_x,
                                           num_blocks_y,
                                           num_blocks_x](
                                              const tt_metal::Program &program,
                                              const std::vector<Buffer *> &input_buffers,
                                              const std::vector<Buffer *> &output_buffers) {
        auto src_dram_buffer_a = input_buffers.at(0);
        auto src_dram_buffer_b = input_buffers.at(1);

        auto dst_dram_buffer = output_buffers.at(0);

        uint32_t num_blocks_read = 0;
        for (int output_idx_y = 0; output_idx_y < num_blocks_y; output_idx_y++) {
            for (int output_idx_x = 0; output_idx_x < num_blocks_x; output_idx_x++) {
                int core_idx_x = num_blocks_read % num_cores_x;
                int core_idx_y = num_blocks_read / num_cores_x;
                CoreCoord core = {(std::size_t)core_idx_x, (std::size_t)core_idx_y};

                {
                    auto &runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
                    runtime_args[0] = src_dram_buffer_a->address();
                    runtime_args[8] = src_dram_buffer_b->address();
                }

                {
                    auto &runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
                    runtime_args[0] = dst_dram_buffer->address();
                }
                num_blocks_read++;
            }
        }
    };

    return {std::move(program), override_runtime_args_callback};
}

namespace ttnn {

namespace operations {

namespace matmul {

operation::ProgramWithCallbacks matmul_multi_core_reuse(
    const Tensor &a, const Tensor &b, Tensor &output, bool bcast_batch) {
    const auto &ashape = a.get_legacy_shape(), bshape = b.get_legacy_shape();

    tt::DataFormat in0_cb_data_format = tt_metal::datatype_to_dataformat_converter(a.get_dtype());
    tt::DataFormat in1_cb_data_format = tt_metal::datatype_to_dataformat_converter(b.get_dtype());
    tt::DataFormat out_cb_data_format = tt_metal::datatype_to_dataformat_converter(output.get_dtype());
    MathFidelity math_fidelity = MathFidelity::HiFi4;

    tt_metal::Buffer *in0_buffer = a.buffer();
    tt_metal::Buffer *in1_buffer = b.buffer();

    ////////////////////////////////////////////////////////////////////////////
    //                      Matmul Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    // NOTE: Only supports matmuls where output is blocks of 16 x 16 tiles (ie. multiples of 16*32 x 16*32)
    // NOTE: Maximum number of tiles in output is 120 * 16^2 = 30,720 (eg. [1, 1, 5120, 6144])
    uint32_t B = get_batch_size(ashape);
    uint32_t Mt = ashape[-2] / TILE_HEIGHT;
    uint32_t Kt = ashape[-1] / TILE_WIDTH;
    uint32_t Nt = bshape[-1] / TILE_WIDTH;
    uint32_t in0_block_w = 2;
    uint32_t out_subblock_h = 4;
    uint32_t out_subblock_w = 2;
    uint32_t per_core_M = 16;
    uint32_t per_core_N = 16;

    TT_FATAL(Mt % per_core_M == 0);
    TT_FATAL(Nt % per_core_N == 0);
    TT_FATAL(Kt % in0_block_w == 0);

    // This should allocate a DRAM buffer on the device
    tt_metal::Device *device = a.device();
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;

    uint32_t num_blocks_total = (Mt / per_core_M) * (Nt / per_core_N);
    TT_FATAL(num_blocks_total <= num_cores_x * num_cores_y);

    ////////////////////////////////////////////////////////////////////////////
    //                      Grayskull Device Setup
    ////////////////////////////////////////////////////////////////////////////
    tt::tt_metal::Shape cshape = output.get_legacy_shape();  // C=A*B, N1MK*11KN->N1MN
    tt_metal::Buffer *out_buffer = output.buffer();
    TT_FATAL(out_buffer != nullptr, "Output buffer should be allocated on device!");

    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////

    return create_program(
        device,
        in0_cb_data_format,
        in1_cb_data_format,
        out_cb_data_format,
        math_fidelity,
        num_cores_x,
        B,
        Mt,
        Nt,
        Kt,
        bcast_batch,
        in0_block_w,
        out_subblock_h,
        out_subblock_w,
        per_core_M,
        per_core_N,
        in0_buffer,
        in1_buffer,
        out_buffer);
}

}  // namespace matmul

}  // namespace operations

}  // namespace ttnn

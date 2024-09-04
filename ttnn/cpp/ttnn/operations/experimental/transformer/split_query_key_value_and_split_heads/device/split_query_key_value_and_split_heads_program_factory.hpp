// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"

namespace ttnn::operations::experimental::transformer::detail {

using namespace tt::constants;
using namespace tt;
using namespace tt_metal;

operation::ProgramWithCallbacks multi_core_split_query_key_value_and_split_heads(const Tensor &a, std::vector<Tensor>& output, CoreCoord compute_with_storage_grid_size) {

    const auto& ashape = a.get_legacy_shape();

    tt_metal::Device *device = a.device();

    tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(a.get_dtype());

    uint32_t single_tile_size = tt_metal::detail::TileSize(cb_data_format);
    tt_metal::Buffer *in0_buffer = a.buffer();
    TT_ASSERT(in0_buffer->size() % single_tile_size == 0);


    ////////////////////////////////////////////////////////////////////////////
    //                      TM Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    uint32_t per_core_tiles = ashape[3] / TILE_WIDTH; // 96
    uint32_t num_tensors = 3;
    uint32_t num_tiles_per_tensor = per_core_tiles / num_tensors; // 32
    uint32_t block_size = 1; // TODO: Play around with different reader and writer block_sizes?
    bool block_size_is_one = block_size == 1;
    uint32_t num_blocks_per_tensor = num_tiles_per_tensor / block_size;

    // Per output tensor args
    // Output shape is: [B, 16, 384, 64] (Q, V heads) or [B, 16, 64, 384] (K heads)
    // For K heads, we write "w_dim" to h_dim instead, but keep nomenclature the same
    uint32_t out_h_tiles = ashape[2] / TILE_HEIGHT;
    uint32_t out_w = 64;
    uint32_t out_w_tiles = out_w / TILE_WIDTH;
    uint32_t out_c = num_tiles_per_tensor / out_w_tiles;
    uint32_t out_HtWt = out_h_tiles * out_w_tiles;
    uint32_t out_CHtWt = out_c * out_HtWt;
    // If block_size_is_one, writer kernel waits differently
    uint32_t writer_num_blocks_per_tensor = block_size_is_one ? 16 : num_blocks_per_tensor;
    uint32_t num_c_per_block = block_size_is_one ? 1 : block_size / out_w_tiles;

    // Parallelize ashape[2] (384 / 32 = 12 tiles) across columns
    // Parallelize ashape[0] (B) across rows
    uint32_t num_cores_x = ashape[2] / TILE_HEIGHT;
    uint32_t num_cores_y = ashape[0];
    TT_ASSERT(num_cores_x <= compute_with_storage_grid_size.x);
    TT_ASSERT(num_cores_y <= compute_with_storage_grid_size.y);
    CoreCoord core_range = {num_cores_x, num_cores_y};


    ////////////////////////////////////////////////////////////////////////////
    //                      Grayskull Device Setup
    ////////////////////////////////////////////////////////////////////////////
    TT_ASSERT((output.size() == 3), "Output vector must be size 3 for split fused qkv!");
    tt_metal::Tensor& q = output[0];
    tt_metal::Tensor& k = output[1];
    tt_metal::Tensor& v = output[2];

    tt_metal::Buffer *q_buffer = q.buffer();
    TT_ASSERT(q_buffer != nullptr, "Output q buffer should be allocated on device!");
    tt_metal::Buffer *k_buffer = k.buffer();
    TT_ASSERT(k_buffer != nullptr, "Output k buffer should be allocated on device!");
    tt_metal::Buffer *v_buffer = v.buffer();
    TT_ASSERT(v_buffer != nullptr, "Output v buffer should be allocated on device!");


    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::Program program = tt_metal::CreateProgram();

    uint32_t start_core_x = 0;
    uint32_t start_core_y = 0;
    uint32_t num_cores_c = core_range.x;
    uint32_t num_cores_r = core_range.y;

    CoreRange all_cores(
        {(std::size_t) start_core_x, (std::size_t) start_core_y},
        {(std::size_t) start_core_x + num_cores_c - 1, (std::size_t) start_core_y + num_cores_r - 1}
    );

    bool in0_is_dram = in0_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool out_is_dram = q_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> reader_compile_time_args = {
            // interleaved accessor args
            (std::uint32_t) in0_is_dram,

            // READER COMPILE TIME ARGS
            (std::uint32_t) block_size, // block_size
            (std::uint32_t) num_blocks_per_tensor, // out_num_blocks_per_tensor
    };
    std::vector<uint32_t> writer_compile_time_args = {
            // interleaved accessor args
            (std::uint32_t) out_is_dram,

            // WRITER COMPILE TIME ARGS
            (std::uint32_t) block_size_is_one,
            (std::uint32_t) block_size, // block_size
            (std::uint32_t) writer_num_blocks_per_tensor, // out_num_blocks_per_tensor
            (std::uint32_t) num_c_per_block, // out_num_c_per_block
            (std::uint32_t) out_w_tiles, // out_w_tiles
            (std::uint32_t) out_h_tiles, // out_h_tiles
            (std::uint32_t) out_HtWt, // out_HtWt
    };

    auto reader_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/transformer/split_query_key_value_and_split_heads/device/kernels/dataflow/reader_tm_tile_layout_create_qkv_heads.cpp",
        all_cores,
        tt_metal::ReaderDataMovementConfig(reader_compile_time_args));
    auto writer_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/transformer/split_query_key_value_and_split_heads/device/kernels/dataflow/writer_tm_tile_layout_create_qkv_heads.cpp",
        all_cores,
        tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    // Dummy compute kernel
    std::vector<uint32_t> compute_args = {num_tiles_per_tensor};
    auto compute_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/compute/transpose_wh.cpp",
        all_cores,
        tt_metal::ComputeConfig{.compile_args = compute_args}
    );

    // Create circular buffers
    // Use cb0 and cb16 for K heads, which uses compute for transpose_wh
    uint32_t src0_cb_index = 0;
    uint32_t cb0_tiles = num_tiles_per_tensor * 2; // double buffer
    uint32_t out_cb_index = 16;
    uint32_t out_cb_tiles = num_tiles_per_tensor;
    // Use cb1 as input and output for Q and V heads
    uint32_t src1_cb_index = 1;
    uint32_t cb1_tiles = num_tiles_per_tensor * 4; // 2 tensors + double buffer
    tt_metal::CircularBufferConfig cb_src0_config = tt_metal::CircularBufferConfig(cb0_tiles * single_tile_size, {{src0_cb_index, cb_data_format}})
		.set_page_size(src0_cb_index, single_tile_size);
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    tt_metal::CircularBufferConfig cb_src1_config = tt_metal::CircularBufferConfig(cb1_tiles * single_tile_size, {{src1_cb_index, cb_data_format}})
		.set_page_size(src1_cb_index, single_tile_size);
    auto cb_src1 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src1_config);

    tt_metal::CircularBufferConfig cb_out_config = tt_metal::CircularBufferConfig(out_cb_tiles * single_tile_size, {{out_cb_index, cb_data_format}})
		.set_page_size(out_cb_index, single_tile_size);
    auto cb_out = tt_metal::CreateCircularBuffer(program, all_cores, cb_out_config);

    for (int core_idx_y = 0; core_idx_y < num_cores_r; core_idx_y++) {
        for (int core_idx_x = 0; core_idx_x < num_cores_c; core_idx_x++) {
            CoreCoord core = {(std::size_t) start_core_x + core_idx_x, (std::size_t) start_core_y + core_idx_y};

            std::vector<uint32_t> reader_runtime_args = {
                (std::uint32_t) in0_buffer->address(), // in0_tensor_addr,
                (core_idx_x + core_idx_y * num_cores_c) * per_core_tiles, // in0_tensor_tile_id
            };
            std::vector<uint32_t> writer_runtime_args = {
                (std::uint32_t) q_buffer->address(), // q_tensor_addr
                (std::uint32_t) k_buffer->address(), // k_tensor_addr
                (std::uint32_t) v_buffer->address(), // v_tensor_addr
                core_idx_x * out_w_tiles + core_idx_y * out_CHtWt, // out_tensor_tile_id
                core_idx_x + core_idx_y * out_CHtWt, // out_tensor_tile_id_with_transpose
            };

            tt_metal::SetRuntimeArgs(program, reader_kernel_id, core, reader_runtime_args);
            tt_metal::SetRuntimeArgs(program, writer_kernel_id, core, writer_runtime_args);
        }
    }

    auto override_runtime_args_callback = [
            reader_kernel_id,
            writer_kernel_id,
            num_cores_r,
            num_cores_c,
            start_core_x,
            start_core_y
        ]
    (
        const Program &program,
        const std::vector<Buffer*>& input_buffers,
        const std::vector<Buffer*>& output_buffers
    ) {

        auto src_dram_buffer = input_buffers.at(0);

        auto dst_dram_buffer_query = output_buffers.at(0);
        auto dst_dram_buffer_key = output_buffers.at(1);
        auto dst_dram_buffer_value = output_buffers.at(2);

        for (int core_idx_y = 0; core_idx_y < num_cores_r; core_idx_y++) {
            for (int core_idx_x = 0; core_idx_x < num_cores_c; core_idx_x++) {
                CoreCoord core = {(std::size_t) start_core_x + core_idx_x, (std::size_t) start_core_y + core_idx_y};

                {
                    auto &runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
                    runtime_args[0] = src_dram_buffer->address();
                }

                {
                    auto &runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
                    runtime_args[0] = dst_dram_buffer_query->address();
                    runtime_args[1] = dst_dram_buffer_key->address();
                    runtime_args[2] = dst_dram_buffer_value->address();
                }
            }
        }
    };

    return {std::move(program), override_runtime_args_callback};
}

operation::ProgramWithCallbacks multi_core_split_query_key_value_and_split_heads_sharded(const Tensor &a, std::vector<Tensor>& output, CoreCoord compute_with_storage_grid_size) {

    tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(a.get_dtype());
    uint32_t single_tile_size = tt_metal::detail::TileSize(cb_data_format);

    ////////////////////////////////////////////////////////////////////////////
    //                      TM Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    auto all_cores = a.shard_spec().value().grid;
    auto bbox = all_cores.bounding_box();
    ShardOrientation shard_orientation = a.shard_spec().value().orientation;
    bool rm = shard_orientation == ShardOrientation::ROW_MAJOR;
    uint32_t num_h_cores = rm ? bbox.end_coord.y + 1 : bbox.end_coord.x + 1;
    uint32_t num_w_cores = rm ? bbox.end_coord.x + 1 : bbox.end_coord.y + 1;
    // tensor shape
    const auto shape = a.get_legacy_shape();
    uint32_t M = shape[2] * shape[0]; // 4608
    uint32_t K = shape[3]; // 3072
    uint32_t Mt = M / TILE_WIDTH;
    uint32_t Kt = K / TILE_WIDTH;
    uint32_t num_tensors = 3;
    uint32_t num_heads_per_tensor = 2;
    // block
    uint32_t block_w = K / num_w_cores; // 384
    uint32_t block_h = M / num_h_cores; // 384
    uint32_t block_wt = block_w / TILE_WIDTH;
    uint32_t block_ht = block_h / TILE_WIDTH;
    uint32_t out_block_w = block_w / num_tensors / num_heads_per_tensor; // 64
    uint32_t out_block_wt = out_block_w / TILE_WIDTH; // 2
    uint32_t out_block_h = block_h * num_heads_per_tensor; // 768
    uint32_t out_block_ht = out_block_h / TILE_WIDTH; // 24
    uint32_t per_core_tiles = block_ht * block_wt;
    uint32_t num_tiles_per_tensor = per_core_tiles / num_tensors;
    // check dims
    TT_ASSERT(M % TILE_WIDTH == 0 && "M must be divisible by tile width.");
    TT_ASSERT(K % TILE_WIDTH == 0 && "K must be divisible by tile width.");
    TT_ASSERT(Kt / num_w_cores == block_wt && "block_w must equal to K / num_cores_w.");
    TT_ASSERT(Mt / num_h_cores == block_ht && "block_h must equal to M / num_cores_h.");
    ////////////////////////////////////////////////////////////////////////////
    //                      Grayskull Device Setup
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::Device *device = a.device();

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    // block size for in0 (tensor a)
    uint32_t in0_CB_size = block_wt * block_ht * single_tile_size;
    // uint32_t im0_CB_size = 2 * single_tile_size;
    uint32_t im0_CB_size = 2 * block_ht * single_tile_size;
    uint32_t out_CB_size = out_block_wt * out_block_ht * single_tile_size;

    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    Program program = CreateProgram();
    // reader compile arg
    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t) num_heads_per_tensor,
        (std::uint32_t) block_ht,
        (std::uint32_t) block_wt,
        (std::uint32_t) out_block_wt,
        (std::uint32_t) block_wt * single_tile_size,
        (std::uint32_t) out_block_wt * single_tile_size,
        (std::uint32_t) num_tiles_per_tensor,
        (std::uint32_t) block_wt * single_tile_size / num_tensors
    };
    auto reader_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/transformer/split_query_key_value_and_split_heads/device/kernels/dataflow/reader_tm_tile_layout_create_qkv_heads_sharded.cpp",
        all_cores,
        tt_metal::ReaderDataMovementConfig(reader_compile_time_args));
    // writer
    std::vector<uint32_t> writer_compile_time_args = {
        (std::uint32_t) num_heads_per_tensor,
        (std::uint32_t) block_ht,
        (std::uint32_t) block_wt,
        (std::uint32_t) out_block_wt,
        (std::uint32_t) block_wt * single_tile_size,
        (std::uint32_t) out_block_wt * single_tile_size,
        (std::uint32_t) num_tiles_per_tensor,
        (std::uint32_t) block_wt * single_tile_size / num_tensors
    };
    auto writer_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/transformer/split_query_key_value_and_split_heads/device/kernels/dataflow/writer_tm_tile_layout_create_qkv_heads_sharded.cpp",
        all_cores,
        tt_metal::WriterDataMovementConfig(writer_compile_time_args));
    // compute kernel
    std::vector<uint32_t> compute_args = {num_tiles_per_tensor};
    auto compute_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/transformer/split_query_key_value_and_split_heads/device/kernels/compute/transpose_wh_sharded.cpp",
        all_cores,
        tt_metal::ComputeConfig{.compile_args = compute_args}
    );

    // Create circular buffers
    // in0 sharded
    auto c_in0_config = CircularBufferConfig(in0_CB_size, {{CB::c_in0, cb_data_format}})
        .set_page_size(CB::c_in0, single_tile_size).set_globally_allocated_address(*a.buffer());
    auto cb_in0_id = CreateCircularBuffer(program, all_cores, c_in0_config);
    // im
    auto c_im0_config = CircularBufferConfig(im0_CB_size, {{CB::c_intermed0, cb_data_format}})
        .set_page_size(CB::c_intermed0, single_tile_size);
    auto cb_im0_id = CreateCircularBuffer(program, all_cores, c_im0_config);
    // q sharded
    auto c_out0_config = CircularBufferConfig(out_CB_size, {{CB::c_out0, cb_data_format}})
        .set_page_size(CB::c_out0, single_tile_size).set_globally_allocated_address(*output[0].buffer());;
    auto cb_out0_id = CreateCircularBuffer( program, all_cores, c_out0_config );
    // k sharded
    auto c_out1_config = CircularBufferConfig(out_CB_size, {{CB::c_out1, cb_data_format}})
        .set_page_size(CB::c_out1, single_tile_size).set_globally_allocated_address(*output[1].buffer());;
    auto cb_out1_id = CreateCircularBuffer( program, all_cores, c_out1_config );
    // v sharded
    auto c_out2_config = CircularBufferConfig(out_CB_size, {{CB::c_out2, cb_data_format}})
        .set_page_size(CB::c_out2, single_tile_size).set_globally_allocated_address(*output[2].buffer());;
    auto cb_out2_id = CreateCircularBuffer( program, all_cores, c_out2_config );

    auto override_runtime_args_callback = [
        cb_in0_id,
        cb_out0_id,
        cb_out1_id,
        cb_out2_id
        ]
    (
        const void* operation,
        Program& program,
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        const std::vector<Tensor>& output_tensors
    ) {
        auto in0_buffer = input_tensors.at(0).buffer();
        auto out0_buffer = output_tensors.at(0).buffer();
        auto out1_buffer = output_tensors.at(1).buffer();
        auto out2_buffer = output_tensors.at(2).buffer();

        UpdateDynamicCircularBufferAddress(program, cb_in0_id, *in0_buffer);
        UpdateDynamicCircularBufferAddress(program, cb_out0_id, *out0_buffer);
        UpdateDynamicCircularBufferAddress(program, cb_out1_id, *out1_buffer);
        UpdateDynamicCircularBufferAddress(program, cb_out2_id, *out2_buffer);
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_args_callback};
}

}  // ttnn::operations::experimental::transformer::detail

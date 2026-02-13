// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "reuse_mcast_2d_descriptor.hpp"

#include <algorithm>
#include <utility>

#include "hostdevcommon/common_values.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "tt-metalium/buffer_types.hpp"

#include "ttnn/operations/matmul/device/utilities/matmul_utilities.hpp"
#include "ttnn/operations/matmul/device/config/matmul_program_config_types.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"
#include "ttnn/operations/compute_throttle_utils.hpp"
#include "ttnn/tensor/shape/shape.hpp"

using ttnn::operations::unary::UnaryOpType;
using ttnn::operations::unary::UnaryWithParam;

using namespace tt;
using namespace tt::constants;

namespace ttnn::prim::matmul_new_detail {

tt::tt_metal::ProgramDescriptor ReuseMcast2DDescriptorFactory::create_descriptor(
    const MatmulParams& operation_attributes,
    const MatmulInputs& tensor_args,
    std::vector<ttnn::Tensor>& tensor_return_value) {
    using namespace tt::tt_metal;
    using tt::tt_metal::TensorMemoryLayout;

    const auto& program_config = std::get<operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig>(
        operation_attributes.program_config.value());

    TT_FATAL(operation_attributes.bcast_batch.has_value(), "bcast_batch should be set");
    TT_FATAL(operation_attributes.compute_kernel_config.has_value(), "compute_kernel_config should be set");

    const auto& a = tensor_args.input_tensors.at(0);
    const auto& b = tensor_args.input_tensors.at(1);
    const auto& bias = tensor_args.optional_input_tensors.at(0);
    auto& output = tensor_return_value.at(0);

    bool bcast_batch = operation_attributes.bcast_batch.value();
    bool transpose_a = operation_attributes.transpose_a;
    bool transpose_b = operation_attributes.transpose_b;
    bool untilize_out = operation_attributes.untilize_out;

    auto fuse_batch = program_config.fuse_batch;
    auto in0_block_w = program_config.in0_block_w;
    auto out_subblock_h = program_config.out_subblock_h;
    auto out_subblock_w = program_config.out_subblock_w;
    auto out_block_h = program_config.out_block_h;
    auto out_block_w = program_config.out_block_w;
    auto per_core_M = program_config.per_core_M;
    auto per_core_N = program_config.per_core_N;
    auto transpose_mcast = program_config.transpose_mcast;
    auto fused_activation = program_config.fused_activation;

    IDevice* device = a.device();

    const auto& ashape = operations::matmul::utilities::get_matmul_tensor_padded_shape(a, transpose_a);
    const auto& bshape = operations::matmul::utilities::get_matmul_tensor_padded_shape(b, transpose_b);
    auto in0_tile = operations::matmul::utilities::get_matmul_tile(a, transpose_a);
    auto in1_tile = operations::matmul::utilities::get_matmul_tile(b, transpose_b);
    auto output_tile = tt::tt_metal::Tile({in0_tile.get_height(), in1_tile.get_width()});

    tt::DataFormat in0_data_format = datatype_to_dataformat_converter(a.dtype());
    tt::DataFormat in1_data_format = datatype_to_dataformat_converter(b.dtype());
    tt::DataFormat output_data_format = datatype_to_dataformat_converter(output.dtype());

    const auto ashape_logical = operations::matmul::utilities::get_matmul_tensor_logical_shape(a, transpose_a);
    const auto in0_last_ktile_w = ashape_logical[-1] % in0_tile.get_width();

    Buffer* in0_buffer = a.buffer();
    Buffer* in1_buffer = b.buffer();
    Buffer* out_buffer = output.buffer();
    Buffer* bias_buffer = nullptr;
    tt::DataFormat bias_data_format = tt::DataFormat::Bfp8_b;
    tt::tt_metal::Tile bias_tile = output_tile;
    if (bias.has_value()) {
        bias_buffer = bias.value().buffer();
        bias_data_format = datatype_to_dataformat_converter(bias.value().dtype());
        bias_tile = bias->tensor_spec().tile();
    }

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), operation_attributes.compute_kernel_config.value());

    uint32_t B = fuse_batch ? 1 : get_batch_size(ashape);
    uint32_t M = operations::matmul::utilities::get_M_dim(ashape, in0_tile, fuse_batch);
    uint32_t K = operations::matmul::utilities::get_K_dim(ashape, in0_tile);
    uint32_t N = operations::matmul::utilities::get_N_dim(bshape, in1_tile);

    // -- Derived parameters --
    bool in0_transpose_tile = in0_tile.get_transpose_of_faces() && in0_tile.get_transpose_within_face();
    bool in1_transpose_tile = in1_tile.get_transpose_of_faces() && in1_tile.get_transpose_within_face();

    TensorMemoryLayout in0_memory_layout = in0_buffer->buffer_layout();
    uint32_t num_blocks = K / in0_block_w;
    bool packer_l1_acc_en = packer_l1_acc && (((bias_buffer != nullptr) && num_blocks > 1) || (num_blocks > 2));

    tt::DataFormat interm0_data_format = packer_l1_acc_en
                                             ? (fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b)
                                             : (fp32_dest_acc_en ? tt::DataFormat::Float32 : output_data_format);

    uint32_t in0_single_tile_size = in0_tile.get_tile_size(in0_data_format);
    uint32_t in1_single_tile_size = in1_tile.get_tile_size(in1_data_format);
    uint32_t bias_single_tile_size = bias_tile.get_tile_size(bias_data_format);
    uint32_t output_single_tile_size = output_tile.get_tile_size(output_data_format);
    uint32_t interm0_single_tile_size = output_tile.get_tile_size(interm0_data_format);

    const bool in0_block_sharded = in0_memory_layout == TensorMemoryLayout::BLOCK_SHARDED;
    const bool in0_height_sharded = in0_memory_layout == TensorMemoryLayout::HEIGHT_SHARDED;
    const bool in0_is_sharded = in0_block_sharded || in0_height_sharded;
    const bool in1_is_sharded = in1_buffer->buffer_layout() == TensorMemoryLayout::WIDTH_SHARDED;
    const bool output_is_sharded = out_buffer->buffer_layout() == TensorMemoryLayout::BLOCK_SHARDED;

    bool do_not_inplace_interm0_out_CB = output_is_sharded && (per_core_M != out_block_h);

    uint32_t in0_block_h = out_block_h;
    uint32_t in1_block_w = out_block_w;
    uint32_t in0_num_blocks_y = per_core_M / out_block_h;
    uint32_t in1_num_blocks_x = per_core_N / out_block_w;
    uint32_t out_num_blocks_x = in1_num_blocks_x;
    uint32_t out_num_blocks_y = in0_num_blocks_y;

    uint32_t in0_block_tiles = out_block_h * in0_block_w;
    uint32_t in0_CB_tiles = in0_block_tiles;
    if (B * num_blocks > 1) {
        in0_CB_tiles *= operations::matmul::utilities::MCAST_INPUT_BUFFERING_DEPTH;
    }
    uint32_t in0_CB_size = in0_CB_tiles * in0_single_tile_size;
    uint32_t in1_block_tiles = out_block_w * in0_block_w;
    uint32_t in1_CB_tiles = in1_block_tiles;
    if (B * num_blocks > 1) {
        in1_CB_tiles *= operations::matmul::utilities::MCAST_INPUT_BUFFERING_DEPTH;
    }
    uint32_t in1_CB_size = in1_CB_tiles * in1_single_tile_size;

    uint32_t out_block_tiles = out_block_h * out_block_w;
    uint32_t out_shard_tiles = per_core_M * per_core_N;
    uint32_t out_CB_tiles = out_block_tiles;
    if (output_is_sharded) {
        out_CB_tiles = out_shard_tiles;
    }
    uint32_t out_CB_size = out_CB_tiles * output_single_tile_size;
    uint32_t interm0_CB_tiles = out_block_tiles;
    uint32_t interm0_CB_size = interm0_CB_tiles * interm0_single_tile_size;

    uint32_t in2_block_tiles = 0;
    uint32_t in0_shard_width_in_tiles = 0;
    uint32_t in0_shard_height_in_tiles = 0;
    if (in0_is_sharded) {
        in0_shard_width_in_tiles = in0_buffer->shard_spec().shape()[1] / in0_tile.get_width();
        in0_shard_height_in_tiles = in0_buffer->shard_spec().shape()[0] / in0_tile.get_height();
        in2_block_tiles = per_core_M * in0_shard_width_in_tiles;
    }
    uint32_t in2_CB_size = in2_block_tiles * in0_single_tile_size;

    uint32_t in3_block_tiles = out_block_w;
    uint32_t in3_CB_size = in3_block_tiles * bias_single_tile_size;

    uint32_t start_core_x = 0;
    uint32_t start_core_y = 0;

    uint32_t num_blocks_y = ((M - 1) / per_core_M) + 1;
    uint32_t num_blocks_x = ((N - 1) / per_core_N) + 1;
    uint32_t num_cores_with_work_c = num_blocks_x;
    uint32_t num_cores_with_work_r = num_blocks_y;
    if (transpose_mcast) {
        std::swap(num_cores_with_work_c, num_cores_with_work_r);
    }

    CoreRange all_cores_with_work(
        {(std::size_t)start_core_x, (std::size_t)start_core_y},
        {(std::size_t)start_core_x + num_cores_with_work_c - 1, (std::size_t)start_core_y + num_cores_with_work_r - 1});

    // -- In0 sharded sender/receiver setup --
    uint32_t num_cores_c = num_cores_with_work_c;
    uint32_t num_cores_r = num_cores_with_work_r;
    uint32_t in0_mcast_receiver_grid_diff_coord_start = 0;
    uint32_t in0_mcast_receiver_grid_diff_coord_end = 0;
    std::vector<uint32_t> in0_mcast_noc_x;
    std::vector<uint32_t> in0_mcast_noc_y;
    uint32_t in0_sender_num_cores_along_width = 0;
    std::optional<CoreRange> in0_mcast_cores_without_work_and_not_in_receiver_grid;

    if (in0_block_sharded) {
        CoreCoord in0_shard_grid = in0_buffer->shard_spec().grid().bounding_box().grid_size();
        in0_sender_num_cores_along_width = transpose_mcast ? in0_shard_grid.y : in0_shard_grid.x;
        if (in0_sender_num_cores_along_width > num_blocks_x) {
            in0_mcast_cores_without_work_and_not_in_receiver_grid =
                transpose_mcast ? CoreRange(
                                      {(std::size_t)start_core_x, (std::size_t)start_core_y + num_blocks_x},
                                      {(std::size_t)start_core_x + num_blocks_y - 1,
                                       (std::size_t)start_core_y + in0_sender_num_cores_along_width - 1})
                                : CoreRange(
                                      {(std::size_t)start_core_x + num_blocks_x, (std::size_t)start_core_y},
                                      {(std::size_t)start_core_x + in0_sender_num_cores_along_width - 1,
                                       (std::size_t)start_core_y + num_blocks_y - 1});
        }

        if (transpose_mcast) {
            in0_mcast_receiver_grid_diff_coord_start = device->worker_core_from_logical_core({0, start_core_y}).y;
            in0_mcast_receiver_grid_diff_coord_end =
                device->worker_core_from_logical_core({0, start_core_y + num_blocks_x - 1}).y;
            in0_mcast_noc_y.reserve(in0_sender_num_cores_along_width);
            for (uint32_t core_idx_y = 0; core_idx_y < in0_sender_num_cores_along_width; ++core_idx_y) {
                in0_mcast_noc_y.push_back(device->worker_core_from_logical_core({0, core_idx_y}).y);
            }
        } else {
            in0_mcast_receiver_grid_diff_coord_start = device->worker_core_from_logical_core({start_core_x, 0}).x;
            in0_mcast_receiver_grid_diff_coord_end =
                device->worker_core_from_logical_core({start_core_x + num_blocks_x - 1, 0}).x;
            in0_mcast_noc_x.reserve(in0_sender_num_cores_along_width);
            for (uint32_t core_idx_x = 0; core_idx_x < in0_sender_num_cores_along_width; ++core_idx_x) {
                in0_mcast_noc_x.push_back(device->worker_core_from_logical_core({core_idx_x, 0}).x);
            }
        }
        if (transpose_mcast) {
            num_cores_r = std::max(num_cores_r, in0_sender_num_cores_along_width);
        } else {
            num_cores_c = std::max(num_cores_c, in0_sender_num_cores_along_width);
        }
    }

    CoreRange all_cores(
        {(std::size_t)start_core_x, (std::size_t)start_core_y},
        {(std::size_t)start_core_x + num_cores_c - 1, (std::size_t)start_core_y + num_cores_r - 1});
    const auto& cores = grid_to_cores(all_cores.start_coord, all_cores.end_coord, true);

    // in0 sender (interleaved) / in1 sender core ranges
    CoreRange in0_sender_interleaved(
        {(std::size_t)start_core_x, (std::size_t)start_core_y},
        {(std::size_t)start_core_x, (std::size_t)start_core_y + num_cores_with_work_r - 1});
    CoreRange in1_sender(
        {(std::size_t)start_core_x, (std::size_t)start_core_y},
        {(std::size_t)start_core_x + num_cores_with_work_c - 1, (std::size_t)start_core_y});

    std::optional<CoreRange> in0_sender_in1_receiver;
    if (num_cores_with_work_r > 1) {
        in0_sender_in1_receiver = {
            {(std::size_t)start_core_x, (std::size_t)start_core_y + 1},
            {(std::size_t)start_core_x, (std::size_t)start_core_y + num_cores_with_work_r - 1}};
    }
    std::optional<CoreRange> in0_receiver_in1_sender;
    if (num_cores_with_work_c > 1) {
        in0_receiver_in1_sender = {
            {(std::size_t)start_core_x + 1, (std::size_t)start_core_y},
            {(std::size_t)start_core_x + num_cores_with_work_c - 1, (std::size_t)start_core_y}};
    }
    if (transpose_mcast) {
        std::swap(in0_sender_interleaved, in1_sender);
        std::swap(in0_sender_in1_receiver, in0_receiver_in1_sender);
    }

    bool split_half = num_cores_with_work_c > 2 && num_cores_with_work_r > 1 && !in0_is_sharded;
    uint32_t half_core = split_half ? (num_cores_with_work_c) / 2 : num_cores_with_work_c - 1;

    std::optional<CoreRange> in0_receiver_in1_receiver_left_half;
    if (num_cores_with_work_c > 1 && num_cores_with_work_r > 1) {
        in0_receiver_in1_receiver_left_half = {
            {(std::size_t)start_core_x + 1, (std::size_t)start_core_y + 1},
            {(std::size_t)start_core_x + half_core, (std::size_t)start_core_y + num_cores_with_work_r - 1}};
    }

    std::set<CoreRange> in0_receiver_interleaved_set;
    std::set<CoreRange> in1_receiver_set;
    if (in0_receiver_in1_sender.has_value()) {
        in0_receiver_interleaved_set.insert(in0_receiver_in1_sender.value());
    }
    if (in0_sender_in1_receiver.has_value()) {
        in1_receiver_set.insert(in0_sender_in1_receiver.value());
    }
    if (in0_receiver_in1_receiver_left_half.has_value()) {
        in0_receiver_interleaved_set.insert(in0_receiver_in1_receiver_left_half.value());
        in1_receiver_set.insert(in0_receiver_in1_receiver_left_half.value());
    }
    CoreRangeSet in0_receiver_interleaved(in0_receiver_interleaved_set);
    CoreRangeSet in1_receiver(in1_receiver_set);

    std::optional<CoreRange> in0_receiver_in1_receiver_interleaved_other_cores;
    if (split_half) {
        in0_receiver_in1_receiver_interleaved_other_cores = {
            {(std::size_t)start_core_x + half_core + 1, (std::size_t)start_core_y + 1},
            {(std::size_t)start_core_x + num_cores_with_work_c - 1,
             (std::size_t)start_core_y + num_cores_with_work_r - 1}};
    }

    // ---- Build ProgramDescriptor ----
    ProgramDescriptor desc;

    // -- Semaphores --
    // We need 4 semaphores: in0 sender/receiver, in1 sender/receiver
    // Use INVALID (0) as initial value
    desc.semaphores.push_back(SemaphoreDescriptor{
        .id = 0, .core_type = CoreType::WORKER, .core_ranges = CoreRangeSet({all_cores}), .initial_value = INVALID});
    desc.semaphores.push_back(SemaphoreDescriptor{
        .id = 1, .core_type = CoreType::WORKER, .core_ranges = CoreRangeSet({all_cores}), .initial_value = INVALID});
    desc.semaphores.push_back(SemaphoreDescriptor{
        .id = 2, .core_type = CoreType::WORKER, .core_ranges = CoreRangeSet({all_cores}), .initial_value = INVALID});
    desc.semaphores.push_back(SemaphoreDescriptor{
        .id = 3, .core_type = CoreType::WORKER, .core_ranges = CoreRangeSet({all_cores}), .initial_value = INVALID});
    uint32_t in0_mcast_sender_semaphore_id = 0;
    uint32_t in0_mcast_receiver_semaphore_id = 1;
    uint32_t in1_mcast_sender_semaphore_id = 2;
    uint32_t in1_mcast_receiver_semaphore_id = 3;

    bool in1_is_dram = in1_buffer->buffer_type() == BufferType::DRAM;

    uint32_t in0_num_subblocks = (out_block_h / out_subblock_h);
    uint32_t in0_block_num_tiles = out_subblock_h * in0_block_w * in0_num_subblocks;

    uint32_t num_dram_banks = 0;
    uint32_t per_core_N_storage = 0;
    if (in1_is_sharded && in1_is_dram) {
        num_dram_banks = device->num_dram_channels();
        per_core_N_storage = (N + num_dram_banks - 1) / num_dram_banks;
    }

    const auto in0_tensor_stride_w = transpose_a ? M : 1;
    const auto in0_tensor_stride_h = transpose_a ? 1 : K;
    const auto in0_tensor_next_block_stride = in0_block_w * in0_tensor_stride_w;
    const auto in0_tensor_next_h_dim_block_stride = in0_block_h * in0_tensor_stride_h;
    const auto in0_tensor_start_tile_id_stride = per_core_M * in0_tensor_stride_h;
    const auto in1_tensor_stride_w = transpose_b ? K : 1;
    const auto in1_tensor_stride_h = transpose_b ? 1 : N;
    const auto in1_tensor_next_block_stride = in0_block_w * in1_tensor_stride_h;
    const auto in1_tensor_next_w_dim_block_stride = in1_block_w * in1_tensor_stride_w;
    const auto in1_tensor_start_tile_id_stride = per_core_N * in1_tensor_stride_w;

    // -- Compile time args for in0 sender --
    std::vector<uint32_t> in0_sender_compile_time_args;
    if (in0_block_sharded) {
        uint32_t num_x = in0_sender_num_cores_along_width;
        uint32_t num_y = 1;
        if (transpose_mcast) {
            std::swap(num_x, num_y);
        }

        in0_sender_compile_time_args = {
            1,
            1,  // core_has_output_block_work, core_in_in0_receiver_mcast_grid
            (uint32_t)in0_block_num_tiles,
            (uint32_t)(in0_block_num_tiles * in0_single_tile_size),
            (uint32_t)in0_last_ktile_w,
            (uint32_t)num_blocks,
            (uint32_t)out_num_blocks_x,
            (uint32_t)out_num_blocks_y,
            in0_mcast_sender_semaphore_id,
            in0_mcast_receiver_semaphore_id,
            (uint32_t)num_blocks_x,
            (uint32_t)num_blocks_x,
            num_x,
            num_y,
            (uint32_t)transpose_mcast,
            in0_shard_width_in_tiles,
            in0_shard_height_in_tiles,
            in0_block_w,
            in0_block_h,
            B,
        };
    } else {
        in0_sender_compile_time_args = {
            (uint32_t)in0_tensor_stride_w,
            (uint32_t)in0_tensor_stride_h,
            (uint32_t)in0_tensor_next_block_stride,
            (uint32_t)in0_tensor_next_h_dim_block_stride,
            in0_block_w,
            in0_block_h,
            in0_block_num_tiles,
            (uint32_t)in0_last_ktile_w,
            0u,
            in0_shard_width_in_tiles,
            in0_shard_height_in_tiles,
            (uint32_t)num_blocks,
            out_num_blocks_x,
            out_num_blocks_y,
            in0_mcast_sender_semaphore_id,
            in0_mcast_receiver_semaphore_id,
            (uint32_t)(num_blocks_x - 1),
            (uint32_t)(num_blocks_x - 1),
            (uint32_t)(M * K),
            B,
            0u,
            0u,
            1u,
            0u,  // sparsity placeholders, bcast_A=true, get_batch_from_reader=false
        };
    }
    in0_sender_compile_time_args.push_back(0u);  // fuse_op all_gather = false
    TensorAccessorArgs(*in0_buffer).append_to(in0_sender_compile_time_args);
    TensorAccessorArgs().append_to(in0_sender_compile_time_args);  // sparsity placeholder

    // -- Compile time args for in1 sender/writer --
    std::vector<uint32_t> in1_sender_writer_compile_time_args = {
        (uint32_t)in1_tensor_stride_w,
        (uint32_t)in1_tensor_stride_h,
        (uint32_t)in1_tensor_next_block_stride,
        (uint32_t)in1_tensor_next_w_dim_block_stride,
        in1_block_w,
        in0_block_w,
        in1_block_w * in0_block_w,
        (uint32_t)num_blocks,
        out_num_blocks_x,
        out_num_blocks_y,
        in1_mcast_sender_semaphore_id,
        in1_mcast_receiver_semaphore_id,
        (uint32_t)(num_blocks_y - 1),
        (uint32_t)(num_blocks_y - 1),
        (uint32_t)(K * N),
        B,
        (uint32_t)bcast_batch,
        0u,
        0u,  // sparsity placeholders
        1u,
        N,
        out_subblock_w,
        out_subblock_h * N,
        out_block_w,
        out_block_h * N,
        out_subblock_w,
        out_subblock_h,
        out_subblock_w * out_subblock_h,
        M * N,
    };
    in1_sender_writer_compile_time_args.push_back(
        bias_buffer != nullptr ? 1u : 0u);              // in3_tensor_stride_w or placeholder
    in1_sender_writer_compile_time_args.push_back(0u);  // fuse_op all_gather = false
    in1_sender_writer_compile_time_args.push_back(0u);  // fuse_op reduce_scatter = false

    TensorAccessorArgs(*in1_buffer).append_to(in1_sender_writer_compile_time_args);
    TensorAccessorArgs().append_to(in1_sender_writer_compile_time_args);  // sparsity placeholder
    TensorAccessorArgs(*out_buffer).append_to(in1_sender_writer_compile_time_args);
    if (bias_buffer != nullptr) {
        TensorAccessorArgs(*bias_buffer).append_to(in1_sender_writer_compile_time_args);
    }
    if (in1_is_sharded && in1_is_dram) {
        in1_sender_writer_compile_time_args.push_back(per_core_N_storage * in0_block_w);
        in1_sender_writer_compile_time_args.push_back(per_core_N_storage * in1_single_tile_size);
    }

    // -- Compile time args for in0 receiver --
    std::vector<uint32_t> in0_receiver_compile_time_args = {
        in0_block_w * in0_block_h,
        (uint32_t)num_blocks,
        out_num_blocks_x,
        out_num_blocks_y,
        in0_mcast_sender_semaphore_id,
        in0_mcast_receiver_semaphore_id,
        B,
        0u,  // batch, get_batch_from_reader=false
    };

    // -- Compile time args for in1 receiver/writer --
    std::vector<uint32_t> in1_receiver_writer_compile_time_args = {
        in1_block_w * in0_block_w,
        (uint32_t)num_blocks,
        out_num_blocks_x,
        out_num_blocks_y,
        in1_mcast_sender_semaphore_id,
        in1_mcast_receiver_semaphore_id,
        B,
        1u,
        N,
        out_subblock_w,
        out_subblock_h * N,
        out_block_w,
        out_block_h * N,
        out_subblock_w,
        out_subblock_h,
        out_subblock_w * out_subblock_h,
        M * N,
    };
    in1_receiver_writer_compile_time_args.push_back(bias_buffer != nullptr ? in1_block_w : 0u);
    in1_receiver_writer_compile_time_args.push_back(0u);  // fuse_op reduce_scatter = false
    TensorAccessorArgs(*out_buffer).append_to(in1_receiver_writer_compile_time_args);

    // -- Kernel defines --
    KernelDescriptor::Defines mm_kernel_defines;
    if (bias_buffer != nullptr) {
        mm_kernel_defines.emplace_back("FUSE_BIAS", "1");
    }
    if (fused_activation.has_value()) {
        if (fused_activation.value().op_type == UnaryOpType::RELU) {
            mm_kernel_defines.emplace_back("PACK_RELU", "1");
        } else {
            using ttnn::operations::unary::utils::get_defines;
            auto map_defs = get_defines(
                fused_activation.value().op_type,
                fused_activation.value().params,
                "ACTIVATION",
                "i",
                dataformat_to_datatype_converter(output_data_format));
            for (auto& [k, v] : map_defs) {
                mm_kernel_defines.emplace_back(k, v);
            }
        }
    }
    if (packer_l1_acc_en) {
        mm_kernel_defines.emplace_back("PACKER_L1_ACC", "1");
    }
    if (fp32_dest_acc_en) {
        mm_kernel_defines.emplace_back("FP32_DEST_ACC_EN", "1");
    }
    if (in1_transpose_tile) {
        mm_kernel_defines.emplace_back("IN1_TRANSPOSE_TILE", "1");
    }
    {
        std::map<std::string, std::string> stagger_defs;
        ttnn::operations::compute_throttle_utils::add_stagger_defines_if_needed(
            device->arch(), cores.size(), stagger_defs);
        ttnn::operations::compute_throttle_utils::throttle_mm_perf(
            device->arch(),
            cores.size(),
            stagger_defs,
            ttnn::get_throttle_level(operation_attributes.compute_kernel_config.value()));
        for (auto& [k, v] : stagger_defs) {
            mm_kernel_defines.emplace_back(k, v);
        }
    }

    KernelDescriptor::Defines in0_sender_sharded_defines;
    KernelDescriptor::Defines in0_sender_interleaved_defines;
    if (in0_receiver_interleaved.num_cores() == 0) {
        in0_sender_interleaved_defines.emplace_back("SKIP_MCAST", "1");
    }
    if (in0_height_sharded) {
        in0_sender_interleaved_defines.emplace_back("IN0_SHARDED", "1");
    }

    KernelDescriptor::Defines in1_sender_writer_defines;
    if (bias_buffer != nullptr) {
        in1_sender_writer_defines.emplace_back("FUSE_BIAS", "1");
    }
    if (in1_receiver.num_cores() == 0) {
        in1_sender_writer_defines.emplace_back("SKIP_MCAST", "1");
    }
    if (in1_is_sharded) {
        if (in1_is_dram) {
            in1_sender_writer_defines.emplace_back("IN1_DRAM_SHARDED", "1");
        } else {
            in1_sender_writer_defines.emplace_back("IN1_SHARDED", "1");
        }
    }
    if (output_is_sharded) {
        in1_sender_writer_defines.emplace_back("OUT_SHARDED", "1");
    }

    KernelDescriptor::Defines in1_receiver_writer_defines;
    if (bias_buffer != nullptr) {
        in1_receiver_writer_defines.emplace_back("FUSE_BIAS", "1");
    }
    if (output_is_sharded) {
        in1_receiver_writer_defines.emplace_back("OUT_SHARDED", "1");
    }

    KernelDescriptor::Defines in1_receiver_writer_other_noc_defines = in1_receiver_writer_defines;

    // Intermediate CB read workaround for Blackhole
    bool in0_needs_intermediate_cb_read = false;
    bool in1_needs_intermediate_cb_read = false;
    if (device->arch() == tt::ARCH::BLACKHOLE) {
        in0_needs_intermediate_cb_read = ((in0_single_tile_size % 64) != 0);
        if (in0_needs_intermediate_cb_read) {
            in0_sender_interleaved_defines.emplace_back("INTERMEDIATE_CB_READ", "1");
        }
        in1_needs_intermediate_cb_read = ((in1_single_tile_size % 64) != 0);
        if (in1_needs_intermediate_cb_read) {
            in1_sender_writer_defines.emplace_back("INTERMEDIATE_CB_READ", "1");
        }
    }

    NOC in0_noc = detail::preferred_noc_for_dram_write(device->arch());
    NOC in1_noc = detail::preferred_noc_for_dram_read(device->arch());
    NOC in0_split_noc = detail::preferred_noc_for_dram_read(device->arch());
    NOC in1_split_noc = detail::preferred_noc_for_dram_write(device->arch());

    // -- Kernels --
    // in0 sender kernel(s)
    KernelDescriptor in0_sender_desc;
    KernelDescriptor in0_sender_no_work_desc;  // for block sharded cores without work
    bool has_in0_no_work_kernel = false;

    if (in0_block_sharded) {
        in0_sender_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/"
            "reader_bmm_tile_layout_in0_sender_receiver_padding_block_sharded.cpp";
        in0_sender_desc.core_ranges = CoreRangeSet({all_cores_with_work});
        in0_sender_desc.compile_time_args = in0_sender_compile_time_args;
        in0_sender_desc.defines = in0_sender_sharded_defines;
        in0_sender_desc.config =
            DataMovementConfigDescriptor{.processor = DataMovementProcessor::RISCV_1, .noc = in0_noc};

        if (in0_mcast_cores_without_work_and_not_in_receiver_grid.has_value()) {
            has_in0_no_work_kernel = true;
            auto no_work_args = in0_sender_compile_time_args;
            no_work_args[0] = 0;  // core_has_output_block_work
            no_work_args[1] = 0;  // core_in_in0_receiver_mcast_grid
            in0_sender_no_work_desc.kernel_source = in0_sender_desc.kernel_source;
            in0_sender_no_work_desc.core_ranges =
                CoreRangeSet({in0_mcast_cores_without_work_and_not_in_receiver_grid.value()});
            in0_sender_no_work_desc.compile_time_args = no_work_args;
            in0_sender_no_work_desc.defines = in0_sender_sharded_defines;
            in0_sender_no_work_desc.config =
                DataMovementConfigDescriptor{.processor = DataMovementProcessor::RISCV_1, .noc = in0_noc};
        }
    } else {
        in0_sender_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_sender_padding.cpp";
        in0_sender_desc.core_ranges = CoreRangeSet({in0_sender_interleaved});
        in0_sender_desc.compile_time_args = in0_sender_compile_time_args;
        in0_sender_desc.defines = in0_sender_interleaved_defines;
        in0_sender_desc.config =
            DataMovementConfigDescriptor{.processor = DataMovementProcessor::RISCV_1, .noc = in0_noc};
    }

    // in1 sender/writer kernel
    KernelDescriptor in1_sender_writer_desc;
    in1_sender_writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in1_sender_writer_padding.cpp";
    in1_sender_writer_desc.core_ranges = CoreRangeSet({in1_sender});
    in1_sender_writer_desc.compile_time_args = in1_sender_writer_compile_time_args;
    in1_sender_writer_desc.defines = in1_sender_writer_defines;
    in1_sender_writer_desc.config =
        DataMovementConfigDescriptor{.processor = DataMovementProcessor::RISCV_0, .noc = in1_noc};

    // in1 receiver/writer kernel
    KernelDescriptor in1_receiver_writer_desc;
    bool has_in1_receiver = in1_receiver.num_cores() > 0;
    if (has_in1_receiver) {
        in1_receiver_writer_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/"
            "reader_bmm_tile_layout_in1_receiver_writer_padding.cpp";
        in1_receiver_writer_desc.core_ranges = in1_receiver;
        in1_receiver_writer_desc.compile_time_args = in1_receiver_writer_compile_time_args;
        in1_receiver_writer_desc.defines = in1_receiver_writer_defines;
        in1_receiver_writer_desc.config =
            DataMovementConfigDescriptor{.processor = DataMovementProcessor::RISCV_0, .noc = in1_noc};
    }

    // in0 receiver kernel (interleaved only)
    KernelDescriptor in0_receiver_desc;
    bool has_in0_receiver = !in0_block_sharded && in0_receiver_interleaved.num_cores() > 0;
    if (has_in0_receiver) {
        in0_receiver_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_receiver.cpp";
        in0_receiver_desc.core_ranges = in0_receiver_interleaved;
        in0_receiver_desc.compile_time_args = in0_receiver_compile_time_args;
        in0_receiver_desc.config =
            DataMovementConfigDescriptor{.processor = DataMovementProcessor::RISCV_1, .noc = in0_noc};
    }

    // Split-noc other-half kernels
    KernelDescriptor in1_receiver_writer_other_noc_desc;
    KernelDescriptor in0_receiver_other_noc_desc;
    bool has_other_noc = in0_receiver_in1_receiver_interleaved_other_cores.has_value();
    if (has_other_noc) {
        in1_receiver_writer_other_noc_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/"
            "reader_bmm_tile_layout_in1_receiver_writer_padding.cpp";
        in1_receiver_writer_other_noc_desc.core_ranges =
            CoreRangeSet({in0_receiver_in1_receiver_interleaved_other_cores.value()});
        in1_receiver_writer_other_noc_desc.compile_time_args = in1_receiver_writer_compile_time_args;
        in1_receiver_writer_other_noc_desc.defines = in1_receiver_writer_other_noc_defines;
        in1_receiver_writer_other_noc_desc.config =
            DataMovementConfigDescriptor{.processor = DataMovementProcessor::RISCV_0, .noc = in1_split_noc};

        in0_receiver_other_noc_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_receiver.cpp";
        in0_receiver_other_noc_desc.core_ranges =
            CoreRangeSet({in0_receiver_in1_receiver_interleaved_other_cores.value()});
        in0_receiver_other_noc_desc.compile_time_args = in0_receiver_compile_time_args;
        in0_receiver_other_noc_desc.config =
            DataMovementConfigDescriptor{.processor = DataMovementProcessor::RISCV_1, .noc = in0_split_noc};
    }

    // Compute kernel
    uint32_t in0_subblock_num_tiles = out_subblock_h * in0_block_w;
    uint32_t in1_num_subblocks_compute = (out_block_w / out_subblock_w);
    uint32_t in1_block_num_tiles = out_subblock_w * in0_block_w * in1_num_subblocks_compute;
    uint32_t in1_per_core_w = out_subblock_w * in1_num_subblocks_compute;
    uint32_t out_subblock_num_tiles = out_subblock_h * out_subblock_w;

    std::vector<uint32_t> compute_kernel_args = {
        in0_block_w,
        in0_num_subblocks,
        in0_block_num_tiles,
        in0_subblock_num_tiles,
        in1_num_subblocks_compute,
        in1_block_num_tiles,
        in1_per_core_w,
        (uint32_t)num_blocks,
        out_num_blocks_x,
        out_num_blocks_y,
        out_subblock_h,
        out_subblock_w,
        out_subblock_num_tiles,
        B,
        out_block_tiles,
        untilize_out ? 1u : 0u,
        0u,  // get_batch_from_reader = false
        in0_transpose_tile ? 1u : 0u,
    };

    KernelDescriptor compute_desc;
    compute_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp";
    compute_desc.core_ranges = CoreRangeSet({all_cores_with_work});
    compute_desc.compile_time_args = compute_kernel_args;
    compute_desc.defines = mm_kernel_defines;
    compute_desc.config = ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .math_approx_mode = math_approx_mode,
    };

    // -- Circular Buffers --
    // CB src0
    {
        CBDescriptor cb;
        cb.total_size = in0_CB_size;
        cb.core_ranges = CoreRangeSet({all_cores});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_0),
            .data_format = in0_data_format,
            .page_size = in0_single_tile_size,
            .tile = TileDescriptor(in0_tile)});
        if (in0_height_sharded) {
            cb.buffer = in0_buffer;
        }
        desc.cbs.push_back(std::move(cb));
    }
    // CB src1
    {
        CBDescriptor cb;
        cb.total_size = in1_CB_size;
        cb.core_ranges = CoreRangeSet({all_cores});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_1),
            .data_format = in1_data_format,
            .page_size = in1_single_tile_size,
            .tile = TileDescriptor(in1_tile)});
        if (in1_is_sharded && !in1_is_dram) {
            cb.buffer = in1_buffer;
        }
        desc.cbs.push_back(std::move(cb));
    }
    // CB src2 (block sharded in0)
    if (in0_block_sharded) {
        desc.cbs.push_back(CBDescriptor{
            .total_size = in2_CB_size,
            .core_ranges = CoreRangeSet({all_cores}),
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_2),
                .data_format = in0_data_format,
                .page_size = in0_single_tile_size,
                .tile = TileDescriptor(in0_tile)}}},
            .buffer = in0_buffer,
        });
        // L1 temp CB
        desc.cbs.push_back(CBDescriptor{
            .total_size = 32 * 2,
            .core_ranges = CoreRangeSet({all_cores}),
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_6),
                .data_format = tt::DataFormat::Float16_b,
                .page_size = 32 * 2}}},
        });
    }
    // CB output / interm0
    {
        uint32_t output_cb_index = tt::CBIndex::c_4;
        uint32_t interm0_cb_index = tt::CBIndex::c_5;
        if (do_not_inplace_interm0_out_CB || (interm0_data_format != output_data_format) ||
            (untilize_out && (in1_num_subblocks_compute > 1))) {
            CBDescriptor out_cb;
            out_cb.total_size = out_CB_size;
            out_cb.core_ranges = CoreRangeSet({all_cores});
            out_cb.format_descriptors.push_back(CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(output_cb_index),
                .data_format = output_data_format,
                .page_size = output_single_tile_size,
                .tile = TileDescriptor(output_tile)});
            if (output_is_sharded) {
                out_cb.buffer = out_buffer;
            }
            desc.cbs.push_back(std::move(out_cb));

            desc.cbs.push_back(CBDescriptor{
                .total_size = interm0_CB_size,
                .core_ranges = CoreRangeSet({all_cores}),
                .format_descriptors = {{CBFormatDescriptor{
                    .buffer_index = static_cast<uint8_t>(interm0_cb_index),
                    .data_format = interm0_data_format,
                    .page_size = interm0_single_tile_size,
                    .tile = TileDescriptor(output_tile)}}},
            });
        } else {
            CBDescriptor cb;
            cb.total_size = out_CB_size;
            cb.core_ranges = CoreRangeSet({all_cores});
            cb.format_descriptors.push_back(CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(output_cb_index),
                .data_format = output_data_format,
                .page_size = output_single_tile_size,
                .tile = TileDescriptor(output_tile)});
            cb.format_descriptors.push_back(CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(interm0_cb_index),
                .data_format = interm0_data_format,
                .page_size = interm0_single_tile_size,
                .tile = TileDescriptor(output_tile)});
            if (output_is_sharded) {
                cb.buffer = out_buffer;
            }
            desc.cbs.push_back(std::move(cb));
        }
    }
    // CB bias
    if (bias_buffer != nullptr) {
        desc.cbs.push_back(CBDescriptor{
            .total_size = in3_CB_size,
            .core_ranges = CoreRangeSet({all_cores}),
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_3),
                .data_format = bias_data_format,
                .page_size = bias_single_tile_size,
                .tile = TileDescriptor(bias_tile)}}},
        });
    }
    // Intermediate CB read
    if (in1_needs_intermediate_cb_read) {
        desc.cbs.push_back(CBDescriptor{
            .total_size = in1_single_tile_size,
            .core_ranges = CoreRangeSet({all_cores}),
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_9),
                .data_format = in1_data_format,
                .page_size = in1_single_tile_size,
                .tile = TileDescriptor(in1_tile)}}},
        });
    }
    if (in0_needs_intermediate_cb_read) {
        desc.cbs.push_back(CBDescriptor{
            .total_size = in0_single_tile_size,
            .core_ranges = CoreRangeSet({all_cores}),
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_8),
                .data_format = in0_data_format,
                .page_size = in0_single_tile_size,
                .tile = TileDescriptor(in0_tile)}}},
        });
    }
    if (in0_transpose_tile) {
        desc.cbs.push_back(CBDescriptor{
            .total_size = in0_CB_size,
            .core_ranges = CoreRangeSet({all_cores}),
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_10),
                .data_format = in0_data_format,
                .page_size = in0_single_tile_size,
                .tile = TileDescriptor(in0_tile)}}},
        });
    }

    // -- Last row/col parameters for padding --
    uint32_t last_per_core_M = M % per_core_M == 0 ? per_core_M : M % per_core_M;
    uint32_t last_per_core_N = N % per_core_N == 0 ? per_core_N : N % per_core_N;
    uint32_t last_out_block_h = last_per_core_M % out_block_h == 0 ? out_block_h : last_per_core_M % out_block_h;
    uint32_t last_out_block_w = last_per_core_N % out_block_w == 0 ? out_block_w : last_per_core_N % out_block_w;
    uint32_t last_out_num_blocks_h = ((last_per_core_M - 1) / out_block_h) + 1;
    uint32_t last_out_num_blocks_w = ((last_per_core_N - 1) / out_block_w) + 1;
    uint32_t last_block_num_nonzero_subblocks_h = ((last_out_block_h - 1) / out_subblock_h) + 1;
    uint32_t last_block_num_nonzero_subblocks_w = ((last_out_block_w - 1) / out_subblock_w) + 1;
    uint32_t last_subblock_of_last_block_h =
        last_out_block_h % out_subblock_h == 0 ? out_subblock_h : last_out_block_h % out_subblock_h;
    uint32_t last_subblock_of_last_block_w =
        last_out_block_w % out_subblock_w == 0 ? out_subblock_w : last_out_block_w % out_subblock_w;
    uint32_t last_block_padded_subblock_tiles_addr_skip =
        output_single_tile_size * (out_subblock_w - last_subblock_of_last_block_w);
    uint32_t last_block_padded_block_tiles_w_skip =
        (out_subblock_w * out_subblock_h) * (out_block_w / out_subblock_w - last_block_num_nonzero_subblocks_w);
    uint32_t last_block_padded_block_tiles_h_skip =
        (out_block_h / out_subblock_h - last_block_num_nonzero_subblocks_h) * (out_block_w * out_subblock_h);

    if (in0_block_sharded) {
        if (in0_noc == NOC::NOC_1) {
            std::swap(in0_mcast_receiver_grid_diff_coord_start, in0_mcast_receiver_grid_diff_coord_end);
        }
    }

    // -- DRAM sharded weights stride params --
    uint32_t worker_core_stride = 0;
    uint32_t storage_core_stride = 0;
    uint32_t curr_worker_core = 0;
    uint32_t curr_storage_core = 0;
    uint32_t vc = 0;

    uint32_t in0_end_idx = num_blocks_y - 1;
    uint32_t in1_end_idx = num_blocks_x - 1;

    // -- Runtime args per core --
    for (const auto& core : cores) {
        CoreCoord left_core = {(std::size_t)start_core_x, (std::size_t)core.y};
        CoreCoord left_core_plus_one = {(std::size_t)start_core_x + 1, (std::size_t)core.y};
        CoreCoord right_core = {(std::size_t)start_core_x + num_cores_with_work_c - 1, (std::size_t)core.y};
        CoreCoord top_core = {(std::size_t)core.x, (std::size_t)start_core_y};
        CoreCoord top_core_plus_one = {(std::size_t)core.x, (std::size_t)start_core_y + 1};
        CoreCoord bottom_core = {(std::size_t)core.x, (std::size_t)start_core_y + num_cores_with_work_r - 1};

        auto left_core_physical = device->worker_core_from_logical_core(left_core);
        auto left_core_plus_one_physical = device->worker_core_from_logical_core(left_core_plus_one);
        auto right_core_physical = device->worker_core_from_logical_core(right_core);
        auto top_core_physical = device->worker_core_from_logical_core(top_core);
        auto top_core_plus_one_physical = device->worker_core_from_logical_core(top_core_plus_one);
        auto bottom_core_physical = device->worker_core_from_logical_core(bottom_core);
        uint32_t in0_idx = core.y - start_core_y;
        uint32_t in1_idx = core.x - start_core_x;

        auto in0_mcast_sender = left_core_physical;
        auto in1_mcast_sender = top_core_physical;

        auto in0_mcast_start = left_core_plus_one_physical;
        auto in0_mcast_end = right_core_physical;
        if (in0_noc == NOC::NOC_1) {
            std::swap(in0_mcast_start, in0_mcast_end);
        }

        auto in1_mcast_start = bottom_core_physical;
        auto in1_mcast_end = top_core_plus_one_physical;
        if (in1_noc == NOC::NOC_0) {
            std::swap(in1_mcast_start, in1_mcast_end);
        }

        if (transpose_mcast) {
            std::swap(in0_idx, in1_idx);
            std::swap(in0_mcast_sender, in1_mcast_sender);
            std::swap(in0_mcast_start, in1_mcast_end);
            std::swap(in0_mcast_end, in1_mcast_start);
        }

        // -- in0 sender runtime args --
        if (in0_block_sharded) {
            uint32_t in0_mcast_receiver_grid_same_coord;
            KernelDescriptor::CoreRuntimeArgs mm_in0_sender_args;
            if (transpose_mcast) {
                in0_mcast_receiver_grid_same_coord = device->worker_core_from_logical_core(core).x;
                mm_in0_sender_args.push_back(core.y);
                mm_in0_sender_args.push_back(in0_mcast_receiver_grid_same_coord);
                mm_in0_sender_args.push_back(in0_mcast_receiver_grid_diff_coord_start);
                mm_in0_sender_args.push_back(in0_mcast_receiver_grid_same_coord);
                mm_in0_sender_args.push_back(in0_mcast_receiver_grid_diff_coord_end);
                mm_in0_sender_args.push_back(in0_mcast_receiver_grid_same_coord);
                mm_in0_sender_args.insert(mm_in0_sender_args.end(), in0_mcast_noc_y.begin(), in0_mcast_noc_y.end());
            } else {
                in0_mcast_receiver_grid_same_coord = device->worker_core_from_logical_core(core).y;
                mm_in0_sender_args.push_back(core.x);
                mm_in0_sender_args.push_back(in0_mcast_receiver_grid_diff_coord_start);
                mm_in0_sender_args.push_back(in0_mcast_receiver_grid_same_coord);
                mm_in0_sender_args.push_back(in0_mcast_receiver_grid_diff_coord_end);
                mm_in0_sender_args.push_back(in0_mcast_receiver_grid_same_coord);
                mm_in0_sender_args.insert(mm_in0_sender_args.end(), in0_mcast_noc_x.begin(), in0_mcast_noc_x.end());
                mm_in0_sender_args.push_back(in0_mcast_receiver_grid_same_coord);
            }
            if (in1_idx < num_blocks_x) {
                in0_sender_desc.runtime_args.emplace_back(core, mm_in0_sender_args);
            } else if (has_in0_no_work_kernel) {
                in0_sender_no_work_desc.runtime_args.emplace_back(core, mm_in0_sender_args);
            }
        } else if (in1_idx == 0) {
            KernelDescriptor::CoreRuntimeArgs mm_in0_sender_args = {
                (uint32_t)in0_buffer->address(),
                (uint32_t)(in0_tensor_start_tile_id_stride * in0_idx),
                (uint32_t)in0_mcast_start.x,
                (uint32_t)in0_mcast_start.y,
                (uint32_t)in0_mcast_end.x,
                (uint32_t)in0_mcast_end.y,
            };
            mm_in0_sender_args.push_back(in0_idx == in0_end_idx ? last_out_block_h : out_block_h);
            mm_in0_sender_args.push_back(0);  // sparsity_addr
            in0_sender_desc.runtime_args.emplace_back(core, mm_in0_sender_args);
        } else if (has_in0_receiver) {
            // in0 receiver
            KernelDescriptor::CoreRuntimeArgs mm_in0_receiver_args = {
                (uint32_t)in0_mcast_sender.x, (uint32_t)in0_mcast_sender.y};
            if (core.x <= half_core || (!transpose_mcast && core.y == start_core_y)) {
                in0_receiver_desc.runtime_args.emplace_back(core, mm_in0_receiver_args);
            } else if (has_other_noc) {
                in0_receiver_other_noc_desc.runtime_args.emplace_back(core, mm_in0_receiver_args);
            }
        }

        if (in0_idx < num_blocks_y && in1_idx < num_blocks_x) {
            if (in0_idx == 0) {
                // in1 sender/writer
                KernelDescriptor::CoreRuntimeArgs mm_in1_sender_writer_args = {
                    (uint32_t)in1_buffer->address(),
                    (uint32_t)(in1_tensor_start_tile_id_stride * in1_idx),
                    (uint32_t)in1_mcast_start.x,
                    (uint32_t)in1_mcast_start.y,
                    (uint32_t)in1_mcast_end.x,
                    (uint32_t)in1_mcast_end.y,
                    0u,  // sparsity_addr
                    (uint32_t)out_buffer->address(),
                    (uint32_t)(in1_idx * per_core_N + in0_idx * per_core_M * N),
                };

                if (in1_idx == in1_end_idx) {
                    mm_in1_sender_writer_args.push_back(last_out_block_w);
                    mm_in1_sender_writer_args.push_back(out_block_h / out_subblock_h);
                    mm_in1_sender_writer_args.push_back(out_subblock_h);
                    mm_in1_sender_writer_args.push_back(0);
                    mm_in1_sender_writer_args.push_back(out_block_w / out_subblock_w);
                    mm_in1_sender_writer_args.push_back(last_block_num_nonzero_subblocks_w);
                    mm_in1_sender_writer_args.push_back(last_subblock_of_last_block_w);
                    mm_in1_sender_writer_args.push_back(last_block_padded_subblock_tiles_addr_skip);
                    mm_in1_sender_writer_args.push_back(last_block_padded_block_tiles_w_skip);
                } else {
                    mm_in1_sender_writer_args.push_back(out_block_w);
                    mm_in1_sender_writer_args.push_back(out_block_h / out_subblock_h);
                    mm_in1_sender_writer_args.push_back(out_subblock_h);
                    mm_in1_sender_writer_args.push_back(0);
                    mm_in1_sender_writer_args.push_back(out_block_w / out_subblock_w);
                    mm_in1_sender_writer_args.push_back(out_block_w / out_subblock_w);
                    mm_in1_sender_writer_args.push_back(out_subblock_w);
                    mm_in1_sender_writer_args.push_back(0);
                    mm_in1_sender_writer_args.push_back(0);
                }
                mm_in1_sender_writer_args.push_back(bias_buffer ? (uint32_t)bias_buffer->address() : 0u);
                mm_in1_sender_writer_args.push_back(bias_buffer ? (uint32_t)(per_core_N * in1_idx) : 0u);
                if (!output_is_sharded) {
                    mm_in1_sender_writer_args.push_back(
                        in1_idx == in1_end_idx ? last_out_num_blocks_w : out_num_blocks_x);
                }

                if (in1_is_sharded && in1_is_dram) {
                    uint32_t num_iter_index = mm_in1_sender_writer_args.size() + 1;
                    vc = vc == 3 ? 0 : vc + 1;
                    mm_in1_sender_writer_args.push_back(vc);
                    uint32_t num_iter = 0;
                    if (curr_storage_core < num_dram_banks) {
                        num_iter++;
                        worker_core_stride = per_core_N_storage - storage_core_stride;
                        mm_in1_sender_writer_args.push_back(storage_core_stride * in1_single_tile_size);
                        mm_in1_sender_writer_args.push_back(worker_core_stride * in1_single_tile_size);
                        mm_in1_sender_writer_args.push_back(curr_storage_core);
                        curr_storage_core += (storage_core_stride + worker_core_stride) / per_core_N_storage;
                        storage_core_stride = (storage_core_stride + worker_core_stride) % per_core_N_storage;
                        uint32_t curr_worker_core_old = curr_worker_core;
                        if (worker_core_stride >= per_core_N) {
                            curr_worker_core += 1;
                        }
                        while (curr_worker_core <= curr_worker_core_old && curr_storage_core < num_dram_banks) {
                            num_iter++;
                            uint32_t stride =
                                std::min(worker_core_stride + (uint32_t)per_core_N_storage, (uint32_t)per_core_N);
                            mm_in1_sender_writer_args.push_back((stride - worker_core_stride) * in1_single_tile_size);
                            mm_in1_sender_writer_args.push_back(curr_storage_core);
                            if (stride >= per_core_N) {
                                curr_worker_core += 1;
                            }
                            storage_core_stride = (stride - worker_core_stride) % per_core_N_storage;
                            curr_storage_core += (stride - worker_core_stride) / per_core_N_storage;
                            worker_core_stride = stride;
                        }
                    }
                    mm_in1_sender_writer_args.insert(mm_in1_sender_writer_args.begin() + num_iter_index, num_iter);
                }
                in1_sender_writer_desc.runtime_args.emplace_back(core, mm_in1_sender_writer_args);
            } else {
                // in1 receiver/writer
                KernelDescriptor::CoreRuntimeArgs mm_in1_receiver_writer_args = {
                    (uint32_t)in1_mcast_sender.x,
                    (uint32_t)in1_mcast_sender.y,
                    (uint32_t)out_buffer->address(),
                    (uint32_t)(in1_idx * per_core_N + in0_idx * per_core_M * N),
                };

                if (in1_idx == in1_end_idx && in0_idx == in0_end_idx) {
                    mm_in1_receiver_writer_args.push_back(out_block_h / out_subblock_h);
                    mm_in1_receiver_writer_args.push_back(last_block_num_nonzero_subblocks_h);
                    mm_in1_receiver_writer_args.push_back(last_subblock_of_last_block_h);
                    mm_in1_receiver_writer_args.push_back(last_block_padded_block_tiles_h_skip);
                    mm_in1_receiver_writer_args.push_back(out_block_w / out_subblock_w);
                    mm_in1_receiver_writer_args.push_back(last_block_num_nonzero_subblocks_w);
                    mm_in1_receiver_writer_args.push_back(last_subblock_of_last_block_w);
                    mm_in1_receiver_writer_args.push_back(last_block_padded_subblock_tiles_addr_skip);
                    mm_in1_receiver_writer_args.push_back(last_block_padded_block_tiles_w_skip);
                } else if (in0_idx == in0_end_idx) {
                    mm_in1_receiver_writer_args.push_back(out_block_h / out_subblock_h);
                    mm_in1_receiver_writer_args.push_back(last_block_num_nonzero_subblocks_h);
                    mm_in1_receiver_writer_args.push_back(last_subblock_of_last_block_h);
                    mm_in1_receiver_writer_args.push_back(last_block_padded_block_tiles_h_skip);
                    mm_in1_receiver_writer_args.push_back(out_block_w / out_subblock_w);
                    mm_in1_receiver_writer_args.push_back(out_block_w / out_subblock_w);
                    mm_in1_receiver_writer_args.push_back(out_subblock_w);
                    mm_in1_receiver_writer_args.push_back(0);
                    mm_in1_receiver_writer_args.push_back(0);
                } else if (in1_idx == in1_end_idx) {
                    mm_in1_receiver_writer_args.push_back(out_block_h / out_subblock_h);
                    mm_in1_receiver_writer_args.push_back(out_block_h / out_subblock_h);
                    mm_in1_receiver_writer_args.push_back(out_subblock_h);
                    mm_in1_receiver_writer_args.push_back(0);
                    mm_in1_receiver_writer_args.push_back(out_block_w / out_subblock_w);
                    mm_in1_receiver_writer_args.push_back(last_block_num_nonzero_subblocks_w);
                    mm_in1_receiver_writer_args.push_back(last_subblock_of_last_block_w);
                    mm_in1_receiver_writer_args.push_back(last_block_padded_subblock_tiles_addr_skip);
                    mm_in1_receiver_writer_args.push_back(last_block_padded_block_tiles_w_skip);
                } else {
                    mm_in1_receiver_writer_args.push_back(out_block_h / out_subblock_h);
                    mm_in1_receiver_writer_args.push_back(out_block_h / out_subblock_h);
                    mm_in1_receiver_writer_args.push_back(out_subblock_h);
                    mm_in1_receiver_writer_args.push_back(0);
                    mm_in1_receiver_writer_args.push_back(out_block_w / out_subblock_w);
                    mm_in1_receiver_writer_args.push_back(out_block_w / out_subblock_w);
                    mm_in1_receiver_writer_args.push_back(out_subblock_w);
                    mm_in1_receiver_writer_args.push_back(0);
                    mm_in1_receiver_writer_args.push_back(0);
                }
                if (!output_is_sharded) {
                    if (in1_idx == in1_end_idx && in0_idx == in0_end_idx) {
                        mm_in1_receiver_writer_args.push_back(last_out_num_blocks_h);
                        mm_in1_receiver_writer_args.push_back(last_out_num_blocks_w);
                    } else if (in0_idx == in0_end_idx) {
                        mm_in1_receiver_writer_args.push_back(last_out_num_blocks_h);
                        mm_in1_receiver_writer_args.push_back(out_num_blocks_x);
                    } else if (in1_idx == in1_end_idx) {
                        mm_in1_receiver_writer_args.push_back(out_num_blocks_y);
                        mm_in1_receiver_writer_args.push_back(last_out_num_blocks_w);
                    } else {
                        mm_in1_receiver_writer_args.push_back(out_num_blocks_y);
                        mm_in1_receiver_writer_args.push_back(out_num_blocks_x);
                    }
                }

                if (has_in1_receiver) {
                    if (core.x <= half_core || (transpose_mcast && core.y == start_core_y)) {
                        in1_receiver_writer_desc.runtime_args.emplace_back(core, mm_in1_receiver_writer_args);
                    } else if (has_other_noc) {
                        in1_receiver_writer_other_noc_desc.runtime_args.emplace_back(core, mm_in1_receiver_writer_args);
                    }
                }
            }
        }
    }

    // -- Push all kernels --
    desc.kernels.push_back(std::move(in0_sender_desc));
    if (has_in0_no_work_kernel) {
        desc.kernels.push_back(std::move(in0_sender_no_work_desc));
    }
    desc.kernels.push_back(std::move(in1_sender_writer_desc));
    if (has_in1_receiver) {
        desc.kernels.push_back(std::move(in1_receiver_writer_desc));
    }
    if (has_in0_receiver) {
        desc.kernels.push_back(std::move(in0_receiver_desc));
    }
    if (has_other_noc) {
        desc.kernels.push_back(std::move(in1_receiver_writer_other_noc_desc));
        desc.kernels.push_back(std::move(in0_receiver_other_noc_desc));
    }
    desc.kernels.push_back(std::move(compute_desc));

    return desc;
}

}  // namespace ttnn::prim::matmul_new_detail

// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dram_sharded_descriptor.hpp"

#include <algorithm>
#include <utility>

#include "hostdevcommon/common_values.hpp"
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/matmul/device/utilities/matmul_utilities.hpp"
#include "ttnn/operations/matmul/device/config/matmul_program_config.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"
#include "ttnn/tensor/shape/shape.hpp"

using namespace tt;

using ttnn::operations::unary::UnaryOpType;
using ttnn::operations::unary::UnaryWithParam;

namespace ttnn::prim::matmul_new_detail {

using dram_sharded_helpers::get_max_page_size_and_num_pages;
using dram_sharded_helpers::get_optimal_dram_bank_to_reader_assignment;
using dram_sharded_helpers::move_common_entries;

tt::tt_metal::ProgramDescriptor DRAMShardedDescriptorFactory::create_descriptor(
    const MatmulParams& operation_attributes,
    const MatmulInputs& tensor_args,
    std::vector<ttnn::Tensor>& tensor_return_value) {
    using namespace tt::tt_metal;

    // ========================================================================
    // Extract tensor information
    // ========================================================================
    const auto& input_tensors = tensor_args.input_tensors;
    const auto& optional_input_tensors = tensor_args.optional_input_tensors;
    const auto& output_tensors = tensor_return_value;

    const auto& a = input_tensors.at(0);
    const auto& b = input_tensors.at(1);
    const auto& bias = optional_input_tensors.at(0);
    const auto& output = output_tensors.at(0);
    const auto& ashape = a.padded_shape();
    const auto& bshape = b.padded_shape();
    auto in0_tile = a.tensor_spec().tile();
    auto in1_tile = b.tensor_spec().tile();
    auto in0_tile_shape = in0_tile.get_tile_shape();
    auto in1_tile_shape = in1_tile.get_tile_shape();
    auto output_tile = tt::tt_metal::Tile({in0_tile.get_tile_shape()[0], in1_tile.get_tile_shape()[1]});

    // CB dataformats
    tt::DataFormat in0_data_format = datatype_to_dataformat_converter(a.dtype());
    tt::DataFormat in1_data_format = datatype_to_dataformat_converter(b.dtype());
    tt::DataFormat output_data_format = datatype_to_dataformat_converter(output.dtype());

    Buffer* bias_buffer = nullptr;
    tt::DataFormat bias_data_format = tt::DataFormat::Bfp8_b;
    tt::tt_metal::Tile bias_tile = output_tile;
    if (bias.has_value()) {
        const auto& c = bias.value();
        TT_FATAL(
            c.storage_type() == StorageType::DEVICE,
            "Bias tensor must be on device, got storage type: {}",
            c.storage_type());
        TT_FATAL(a.device() == c.device(), "Operands to matmul need to be on the same device!");
        TT_FATAL(c.buffer() != nullptr, "Operands to matmul need to be allocated in buffers on device!");
        bias_buffer = c.buffer();
        bias_data_format = datatype_to_dataformat_converter(c.dtype());
        bias_tile = c.tensor_spec().tile();
    }

    IDevice* device = a.device();

    TT_FATAL(
        a.shard_spec().has_value() && output.shard_spec().has_value(), "Both input A and output must have shard specs");
    CoreRangeSet input_all_storage_cores = a.shard_spec().value().grid;
    CoreRangeSet output_all_storage_cores = output.shard_spec().value().grid;

    Buffer* in0_buffer = a.buffer();
    Buffer* in1_buffer = b.buffer();

    uint32_t in0_single_tile_size = in0_tile.get_tile_size(in0_data_format);
    uint32_t in1_single_tile_size = in1_tile.get_tile_size(in1_data_format);

    TT_FATAL(
        in0_buffer->size() % in0_single_tile_size == 0,
        "Input A buffer size ({}) must be divisible by single tile size ({})",
        in0_buffer->size(),
        in0_single_tile_size);
    TT_FATAL(
        in1_buffer->size() % in1_single_tile_size == 0,
        "Input B buffer size ({}) must be divisible by single tile size ({})",
        in1_buffer->size(),
        in1_single_tile_size);
    TT_FATAL(
        ashape[-1] == bshape[-2],
        "Dimension K (A.shape[-1] = {}, B.shape[-2] = {}) must match for matmul",
        ashape[-1],
        bshape[-2]);
    TT_FATAL(
        ashape[-2] % in0_tile_shape[0] == 0,
        "A.shape[-2] ({}) must be divisible by tile shape[0] ({})",
        ashape[-2],
        in0_tile_shape[0]);
    TT_FATAL(
        ashape[-1] % in0_tile_shape[1] == 0,
        "A.shape[-1] ({}) must be divisible by tile shape[1] ({})",
        ashape[-1],
        in0_tile_shape[1]);
    TT_FATAL(
        bshape[-2] % in1_tile_shape[0] == 0,
        "B.shape[-2] ({}) must be divisible by tile shape[0] ({})",
        bshape[-2],
        in1_tile_shape[0]);
    TT_FATAL(
        bshape[-1] % in1_tile_shape[1] == 0,
        "B.shape[-1] ({}) must be divisible by tile shape[1] ({})",
        bshape[-1],
        in1_tile_shape[1]);

    const auto& compute_kernel_config = operation_attributes.compute_kernel_config.value();
    const auto& program_config = std::get<operations::matmul::MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig>(
        operation_attributes.program_config.value());
    const auto& in0_block_w = program_config.in0_block_w;
    const auto& per_core_M = program_config.per_core_M;
    const auto& per_core_N_storage = program_config.per_core_N;
    const auto& fused_activation = program_config.fused_activation;
    const auto& untilize_out = operation_attributes.untilize_out;

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);

    uint32_t B = 1;
    uint32_t K = ashape[-1] / in0_tile_shape[1];
    uint32_t N = bshape[-1] / in1_tile_shape[1];
    uint32_t in0_last_ktile_w = a.logical_shape()[-1] % in0_tile_shape[1];

    TT_FATAL(K % in0_block_w == 0, "Kt ({}) must be divisible by in0_block_w ({})", K, in0_block_w);

    Buffer* out_buffer = output.buffer();
    TT_FATAL(out_buffer != nullptr, "Output buffer should be allocated on device!");

    // ========================================================================
    // Core setup and parameter computation
    // ========================================================================
    bool in1_transpose_tile = in1_tile.get_transpose_of_faces() && in1_tile.get_transpose_within_face();
    TT_FATAL(
        in1_buffer->shard_spec().orientation() == ShardOrientation::ROW_MAJOR, "Only ROW_MAJOR sharding is supported");

    uint32_t start_core_x = 0;
    uint32_t start_core_y = 0;
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_mcast_cores = compute_with_storage_grid_size.x * compute_with_storage_grid_size.y;

    CoreCoord top_left_core = {(std::size_t)start_core_x, (std::size_t)start_core_y};
    CoreCoord bottom_right_core = {
        (std::size_t)start_core_x + compute_with_storage_grid_size.x - 1,
        (std::size_t)start_core_y + compute_with_storage_grid_size.y - 1};
    auto top_left_core_physical = device->worker_core_from_logical_core(top_left_core);
    auto bottom_right_core_physical = device->worker_core_from_logical_core(bottom_right_core);

    NOC in0_noc = detail::preferred_noc_for_dram_write(device->arch());
    NOC in1_noc = detail::preferred_noc_for_dram_read(device->arch());

    CoreCoord start_core_noc = top_left_core_physical;
    CoreCoord end_core_noc = bottom_right_core_physical;
    if (in0_noc == NOC::NOC_1) {
        std::swap(start_core_noc, end_core_noc);
    }

    // Get the optimal DRAM bank to reader assignment
    std::vector<CoreCoord> all_worker_cores_ordered;
    CoreRangeSet all_worker_cores;
    get_optimal_dram_bank_to_reader_assignment(device, all_worker_cores_ordered, all_worker_cores, in1_noc);

    uint32_t num_dram_banks = all_worker_cores_ordered.size();

    // Remove cores assigned to padding-only DRAM banks
    uint32_t in1_shard_width_tiles = in1_buffer->shard_spec().shape()[1] / in1_tile.get_tile_shape()[1];
    uint32_t in1_tensor_padded_width_tiles = in1_shard_width_tiles * num_dram_banks;

    if (in1_tensor_padded_width_tiles > N) {
        uint32_t padding_width_tiles = in1_tensor_padded_width_tiles - N;
        uint32_t only_padding_banks = padding_width_tiles / in1_shard_width_tiles;
        TT_FATAL(
            only_padding_banks < all_worker_cores_ordered.size(),
            "Padding banks count {} must be less than workers count {}",
            only_padding_banks,
            all_worker_cores_ordered.size());
        for (uint32_t i = 0; i < only_padding_banks; ++i) {
            all_worker_cores_ordered.pop_back();
        }
        std::set<CoreRange> new_workers_set;
        for (const auto& worker_core : all_worker_cores_ordered) {
            new_workers_set.insert(CoreRange(worker_core));
        }
        all_worker_cores = CoreRangeSet(new_workers_set);
        num_dram_banks = all_worker_cores_ordered.size();
    }

    uint32_t per_core_N_compute = div_up(N, num_dram_banks);
    uint32_t per_core_N_in1_sender = per_core_N_compute;

    // Subblock parameters
    auto subblock_hw = operations::matmul::bmm_op_utils::get_matmul_subblock_params(
        per_core_M, per_core_N_compute, false, false, fp32_dest_acc_en);
    auto out_subblock_h = std::get<0>(subblock_hw);
    auto out_subblock_w = std::get<1>(subblock_hw);

    uint32_t max_subblock_w = fp32_dest_acc_en ? 4 : 8;
    if (out_subblock_h == 1 and out_subblock_w < max_subblock_w) {
        uint32_t num_subblock_w_per_core_N = per_core_N_compute / out_subblock_w;
        uint32_t num_iter = max_subblock_w - out_subblock_w;
        uint32_t new_out_subblock_w = out_subblock_w;
        uint32_t preferred_out_subblock_w = out_subblock_w;

        for (uint32_t i = 0; i < num_iter; ++i) {
            new_out_subblock_w += 1;
            uint32_t new_num_subblock_w_per_core_N = (per_core_N_compute + new_out_subblock_w - 1) / new_out_subblock_w;
            if (new_num_subblock_w_per_core_N < num_subblock_w_per_core_N) {
                num_subblock_w_per_core_N = new_num_subblock_w_per_core_N;
                preferred_out_subblock_w = new_out_subblock_w;
            }
        }
        out_subblock_w = preferred_out_subblock_w;
        per_core_N_compute = out_subblock_w * num_subblock_w_per_core_N;
    }

    uint32_t num_blocks = K / in0_block_w;
    bool packer_l1_acc_en = packer_l1_acc && num_blocks > 1;

    tt::DataFormat interm0_data_format = packer_l1_acc_en
                                             ? (fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b)
                                             : (fp32_dest_acc_en ? tt::DataFormat::Float32 : output_data_format);

    uint32_t bias_single_tile_size = bias_tile.get_tile_size(bias_data_format);
    uint32_t output_single_tile_size = output_tile.get_tile_size(output_data_format);
    uint32_t interm0_single_tile_size = output_tile.get_tile_size(interm0_data_format);

    // CB sizes
    uint32_t in0_block_tiles = per_core_M * in0_block_w;
    uint32_t in0_CB_tiles = in0_block_tiles;
    if (B * num_blocks > 1) {
        in0_CB_tiles = in0_CB_tiles * 2;  // double buffer
    }
    uint32_t in0_CB_size = in0_CB_tiles * in0_single_tile_size;

    uint32_t in1_block_tiles = per_core_N_in1_sender * in0_block_w;
    uint32_t in1_CB_tiles = in1_block_tiles;
    if (B * num_blocks > 1) {
        in1_CB_tiles = in1_CB_tiles * 3;  // triple buffer
    }
    uint32_t in1_CB_size = in1_CB_tiles * in1_single_tile_size;

    uint32_t out_block_tiles = per_core_M * per_core_N_compute;
    uint32_t out_CB_tiles = out_block_tiles;
    uint32_t out_CB_size = out_CB_tiles * output_single_tile_size;
    uint32_t interm0_CB_size = out_CB_tiles * interm0_single_tile_size;

    uint32_t out_reshard_block_tiles = per_core_M * per_core_N_storage;
    uint32_t out_reshard_CB_size = out_reshard_block_tiles * output_single_tile_size;

    uint32_t in0_shard_width_in_tiles = in0_buffer->shard_spec().shape()[1] / in0_tile.get_tile_shape()[1];
    uint32_t in2_block_tiles = per_core_M * in0_shard_width_in_tiles;
    uint32_t in2_CB_size = in2_block_tiles * in0_single_tile_size;

    uint32_t in3_block_tiles = per_core_N_in1_sender;
    uint32_t in3_CB_size = in3_block_tiles * bias_single_tile_size;

    // Page sizes for DRAM reads
    uint32_t in1_buffer_page_size, in1_buffer_num_pages;
    get_max_page_size_and_num_pages(
        device, in1_block_tiles, in1_single_tile_size, in1_buffer_page_size, in1_buffer_num_pages);

    uint32_t bias_buffer_page_size, bias_buffer_num_pages;
    get_max_page_size_and_num_pages(
        device, in3_block_tiles, bias_single_tile_size, bias_buffer_page_size, bias_buffer_num_pages);

    uint32_t num_worker_cores = num_dram_banks;

    // ========================================================================
    // Core partitioning: mcast senders, receivers, idle
    // ========================================================================
    std::vector<CoreCoord> input_all_storage_cores_vec = corerange_to_cores(input_all_storage_cores);
    std::vector<CoreCoord> all_worker_cores_vec = corerange_to_cores(all_worker_cores);
    std::vector<CoreCoord> storage_worker_common;
    move_common_entries(input_all_storage_cores_vec, all_worker_cores_vec, storage_worker_common);

    std::vector<CoreRange> input_all_storage_cores_range;
    input_all_storage_cores_range.reserve(input_all_storage_cores_vec.size());
    std::transform(
        input_all_storage_cores_vec.begin(),
        input_all_storage_cores_vec.end(),
        std::back_inserter(input_all_storage_cores_range),
        [](const CoreCoord& coord) { return CoreRange(coord); });

    std::vector<CoreRange> all_worker_cores_range;
    all_worker_cores_range.reserve(all_worker_cores_vec.size());
    std::transform(
        all_worker_cores_vec.begin(),
        all_worker_cores_vec.end(),
        std::back_inserter(all_worker_cores_range),
        [](const CoreCoord& coord) { return CoreRange(coord); });

    std::set<CoreRange> input_all_storage_cores_set(
        input_all_storage_cores_range.begin(), input_all_storage_cores_range.end());
    std::set<CoreRange> all_worker_cores_set(all_worker_cores_range.begin(), all_worker_cores_range.end());
    CoreRangeSet mcast_senders = CoreRangeSet(input_all_storage_cores_set);
    CoreRangeSet mcast_receivers = CoreRangeSet(all_worker_cores_set);

    // All cores = senders ∪ receivers
    std::set<CoreRange> all_cores_set;
    all_cores_set.insert(mcast_senders.ranges().begin(), mcast_senders.ranges().end());
    all_cores_set.insert(mcast_receivers.ranges().begin(), mcast_receivers.ranges().end());
    CoreRangeSet all_cores = CoreRangeSet(all_cores_set);

    // Grid bounding box
    CoreRange bounding_box = all_cores.bounding_box();
    std::set<CoreRange> bounding_box_set;
    bounding_box_set.insert(bounding_box);
    CoreRangeSet all_cores_in_rect_grid(bounding_box_set);
    std::vector<CoreCoord> all_cores_in_rect_grid_vec = corerange_to_cores(all_cores_in_rect_grid);

    // ========================================================================
    // Build ProgramDescriptor
    // ========================================================================
    ProgramDescriptor desc;

    // -- Semaphores --
    CoreCoord first_core = all_cores_in_rect_grid_vec[0];

    auto in0_mcast_sender_semaphore_id = desc.find_available_semaphore_id(first_core, CoreType::WORKER).value();
    desc.semaphores.push_back(SemaphoreDescriptor{
        .id = in0_mcast_sender_semaphore_id, .core_ranges = all_cores_in_rect_grid, .initial_value = INVALID});

    auto in0_mcast_receiver_semaphore_id = desc.find_available_semaphore_id(first_core, CoreType::WORKER).value();
    desc.semaphores.push_back(SemaphoreDescriptor{
        .id = in0_mcast_receiver_semaphore_id, .core_ranges = all_cores_in_rect_grid, .initial_value = INVALID});

    auto in0_mcast_sender_valid_semaphore_id = desc.find_available_semaphore_id(first_core, CoreType::WORKER).value();
    desc.semaphores.push_back(SemaphoreDescriptor{
        .id = in0_mcast_sender_valid_semaphore_id, .core_ranges = all_cores_in_rect_grid, .initial_value = VALID});

    // -- Compile-time args --
    uint32_t in0_num_subblocks = (per_core_M / out_subblock_h);
    uint32_t in0_block_num_tiles = out_subblock_h * in0_block_w * in0_num_subblocks;

    uint32_t num_blocks_per_shard = num_blocks / input_all_storage_cores_vec.size();
    if (per_core_M > 1) {
        TT_FATAL(
            num_blocks_per_shard == 1,
            "currently not support per_core_M larger than 1, while split one shard into multiple blocks (per_core_M "
            "{}, num_blocks_per_shard {})",
            per_core_M,
            num_blocks_per_shard);
    }

    std::vector<uint32_t> in0_sender_compile_time_args = {
        (std::uint32_t)in0_block_num_tiles,                         // in0_block_num_tiles
        (std::uint32_t)in0_block_num_tiles * in0_single_tile_size,  // in0_block_size_bytes
        (std::uint32_t)in0_last_ktile_w,                            // in0_last_ktile_w
        // in0 mcast args
        (std::uint32_t)in0_mcast_sender_semaphore_id,
        (std::uint32_t)in0_mcast_receiver_semaphore_id,
        (std::uint32_t)num_worker_cores,  // in0_mcast_num_dests
        (std::uint32_t)num_mcast_cores,   // in0_mcast_num_cores
        // block
        (std::uint32_t)num_blocks,
        // mcast noc coords
        (std::uint32_t)start_core_noc.x,
        (std::uint32_t)start_core_noc.y,
        (std::uint32_t)end_core_noc.x,
        (std::uint32_t)end_core_noc.y,
        // semaphore valid
        (std::uint32_t)in0_mcast_sender_valid_semaphore_id,
        //
        (std::uint32_t)num_blocks_per_shard};

    std::vector<uint32_t> in1_sender_writer_compile_time_args = {
        (std::uint32_t)in1_buffer_page_size,
        (std::uint32_t)in1_buffer_num_pages,
        // in1 block args
        (std::uint32_t)per_core_N_in1_sender,                // in1_block_w
        (std::uint32_t)per_core_N_in1_sender * in0_block_w,  // in1_block_num_tiles
        // in0/in1 common args
        (std::uint32_t)num_blocks,                                    // num_blocks
        (std::uint32_t)out_block_tiles,                               // out_block_num_tiles
        (std::uint32_t)per_core_N_compute * output_single_tile_size,  // out_tensor_stride_w_bytes
        (std::uint32_t)per_core_N_storage * output_single_tile_size,  // out_reshard_tensor_stride_w_bytes
        (std::uint32_t)per_core_M};
    if (bias_buffer != nullptr) {
        in1_sender_writer_compile_time_args.push_back(bias_buffer_page_size);
        in1_sender_writer_compile_time_args.push_back(bias_buffer_num_pages);
        in1_sender_writer_compile_time_args.push_back((std::uint32_t)1);
    }

    uint32_t in0_subblock_num_tiles = out_subblock_h * in0_block_w;
    uint32_t in1_num_subblocks = (per_core_N_compute / out_subblock_w);
    uint32_t in1_per_core_w = per_core_N_in1_sender;
    uint32_t out_subblock_num_tiles = out_subblock_h * out_subblock_w;

    std::vector<uint32_t> compute_kernel_args = {
        in0_block_w,             // in0_block_w
        in0_num_subblocks,       // in0_num_subblocks
        in0_block_num_tiles,     // in0_block_num_tiles
        in0_subblock_num_tiles,  // in0_subblock_num_tiles

        in1_num_subblocks,  // in1_num_subblocks
        in1_block_tiles,    // in1_block_num_tiles
        in1_per_core_w,     // in1_per_core_w

        num_blocks,  // num_blocks
        1,           // out_num_blocks_x
        1,           // out_num_blocks_y

        out_subblock_h,          // out_subblock_h
        out_subblock_w,          // out_subblock_w
        out_subblock_num_tiles,  // out_subblock_num_tiles
        B,                       // batch
        out_block_tiles,         // out_block_num_tiles
        untilize_out ? 1u : 0u,  // untilize_out
        0u,                      // get_batch_from_reader
        0u,                      // in0_transpose_tile
    };

    // -- Kernel defines --
    KernelDescriptor::Defines compute_defines;
    KernelDescriptor::Defines in0_sender_defines;
    KernelDescriptor::Defines in1_sender_writer_defines;

    if (bias_buffer != nullptr) {
        compute_defines.emplace_back("FUSE_BIAS", "1");
        in1_sender_writer_defines.emplace_back("FUSE_BIAS", "1");
    }
    if (fused_activation.has_value()) {
        if (fused_activation.value().op_type == UnaryOpType::RELU) {
            compute_defines.emplace_back("PACK_RELU", "1");
        } else {
            using ttnn::operations::unary::utils::get_defines;
            auto activation_defines = get_defines(
                fused_activation.value().op_type,
                fused_activation.value().params,
                "ACTIVATION",
                "i",
                dataformat_to_datatype_converter(output_data_format));
            for (const auto& [k, v] : activation_defines) {
                compute_defines.emplace_back(k, v);
            }
        }
    }
    if (packer_l1_acc_en) {
        compute_defines.emplace_back("PACKER_L1_ACC", "1");
    }
    if (fp32_dest_acc_en) {
        compute_defines.emplace_back("FP32_DEST_ACC_EN", "1");
    }
    in1_sender_writer_defines.emplace_back("OUT_SHARDED", "1");
    in1_sender_writer_defines.emplace_back("SKIP_MCAST", "1");
    compute_defines.emplace_back("MATMUL_DRAM_SHARDED", "1");
    if (in1_transpose_tile) {
        compute_defines.emplace_back("IN1_TRANSPOSE_TILE", "1");
    }

    // -- Circular Buffers --
    // CB0: src0 (in0)
    {
        CBDescriptor cb;
        cb.total_size = in0_CB_size;
        cb.core_ranges = all_cores_in_rect_grid;
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_0),
            .data_format = in0_data_format,
            .page_size = in0_single_tile_size,
            .tile = TileDescriptor(in0_tile),
        });
        desc.cbs.push_back(std::move(cb));
    }
    // CB1: src1 (in1)
    {
        CBDescriptor cb;
        cb.total_size = in1_CB_size;
        cb.core_ranges = all_cores_in_rect_grid;
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_1),
            .data_format = in1_data_format,
            .page_size = in1_single_tile_size,
            .tile = TileDescriptor(in1_tile),
        });
        desc.cbs.push_back(std::move(cb));
    }
    // CB2: sharded in0 (backed by in0_buffer)
    {
        CBDescriptor cb;
        cb.total_size = in2_CB_size;
        cb.core_ranges = all_cores_in_rect_grid;
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_2),
            .data_format = in0_data_format,
            .page_size = in0_single_tile_size,
            .tile = TileDescriptor(in0_tile),
        });
        cb.buffer = in0_buffer;
        desc.cbs.push_back(std::move(cb));
    }
    // CB4 & CB5: output and interm0
    {
        uint32_t output_cb_index = tt::CBIndex::c_4;
        uint32_t interm0_cb_index = tt::CBIndex::c_5;

        if ((interm0_data_format != output_data_format) || (untilize_out && (in1_num_subblocks > 1))) {
            // Separate output and interm0 CBs
            {
                CBDescriptor cb;
                cb.total_size = out_CB_size;
                cb.core_ranges = all_cores_in_rect_grid;
                cb.format_descriptors.push_back(CBFormatDescriptor{
                    .buffer_index = static_cast<uint8_t>(output_cb_index),
                    .data_format = output_data_format,
                    .page_size = output_single_tile_size,
                    .tile = TileDescriptor(output_tile),
                });
                desc.cbs.push_back(std::move(cb));
            }
            {
                CBDescriptor cb;
                cb.total_size = interm0_CB_size;
                cb.core_ranges = all_cores_in_rect_grid;
                cb.format_descriptors.push_back(CBFormatDescriptor{
                    .buffer_index = static_cast<uint8_t>(interm0_cb_index),
                    .data_format = interm0_data_format,
                    .page_size = interm0_single_tile_size,
                    .tile = TileDescriptor(output_tile),
                });
                desc.cbs.push_back(std::move(cb));
            }
        } else {
            // Shared buffer for output and interm0
            CBDescriptor cb;
            cb.total_size = out_CB_size;
            cb.core_ranges = all_cores_in_rect_grid;
            cb.format_descriptors.push_back(CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(output_cb_index),
                .data_format = output_data_format,
                .page_size = output_single_tile_size,
                .tile = TileDescriptor(output_tile),
            });
            cb.format_descriptors.push_back(CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(interm0_cb_index),
                .data_format = interm0_data_format,
                .page_size = interm0_single_tile_size,
                .tile = TileDescriptor(output_tile),
            });
            desc.cbs.push_back(std::move(cb));
        }
    }
    // CB6: output reshard (backed by out_buffer)
    {
        CBDescriptor cb;
        cb.total_size = out_reshard_CB_size;
        cb.core_ranges = all_cores_in_rect_grid;
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_6),
            .data_format = output_data_format,
            .page_size = output_single_tile_size,
            .tile = TileDescriptor(output_tile),
        });
        cb.buffer = out_buffer;
        desc.cbs.push_back(std::move(cb));
    }
    // CB3: bias (if present)
    if (bias_buffer != nullptr) {
        CBDescriptor cb;
        cb.total_size = in3_CB_size;
        cb.core_ranges = all_cores_in_rect_grid;
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_3),
            .data_format = bias_data_format,
            .page_size = bias_single_tile_size,
            .tile = TileDescriptor(bias_tile),
        });
        desc.cbs.push_back(std::move(cb));
    }

    // -- Kernel Descriptors --
    KernelDescriptor in0_sender_desc;
    in0_sender_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/"
        "reader_bmm_tile_layout_in0_sender_dram_sharded.cpp";
    in0_sender_desc.core_ranges = all_cores_in_rect_grid;
    in0_sender_desc.compile_time_args = in0_sender_compile_time_args;
    in0_sender_desc.defines = in0_sender_defines;
    in0_sender_desc.config = DataMovementConfigDescriptor{
        .processor = DataMovementProcessor::RISCV_1,
        .noc = in0_noc,
    };

    KernelDescriptor in1_sender_writer_desc;
    in1_sender_writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/"
        "reader_bmm_tile_layout_in1_sender_dram_sharded.cpp";
    in1_sender_writer_desc.core_ranges = all_cores_in_rect_grid;
    in1_sender_writer_desc.compile_time_args = in1_sender_writer_compile_time_args;
    in1_sender_writer_desc.defines = in1_sender_writer_defines;
    in1_sender_writer_desc.config = DataMovementConfigDescriptor{
        .processor = DataMovementProcessor::RISCV_0,
        .noc = in1_noc,
    };

    KernelDescriptor compute_desc;
    compute_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/"
        "bmm_large_block_zm_fused_bias_activation.cpp";
    compute_desc.core_ranges = all_cores_in_rect_grid;
    compute_desc.compile_time_args = compute_kernel_args;
    compute_desc.defines = compute_defines;
    compute_desc.config = ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .math_approx_mode = math_approx_mode,
    };

    // ========================================================================
    // Runtime args
    // ========================================================================

    // Build NOC coordinates for mcast senders
    std::vector<uint32_t> in0_mcast_sender_noc_x;
    std::vector<uint32_t> in0_mcast_sender_noc_y;
    std::vector<CoreCoord> mcast_senders_coords = corerange_to_cores(mcast_senders);
    std::sort(mcast_senders_coords.begin(), mcast_senders_coords.end(), [](const CoreCoord& a, const CoreCoord& b) {
        if (a.y != b.y) {
            return a.y < b.y;
        }
        return a.x < b.x;
    });
    in0_mcast_sender_noc_x.reserve(mcast_senders_coords.size());
    for (auto core : mcast_senders_coords) {
        in0_mcast_sender_noc_x.push_back((std::uint32_t)device->worker_core_from_logical_core(core).x);
    }
    in0_mcast_sender_noc_y.reserve(mcast_senders_coords.size());
    for (auto core : mcast_senders_coords) {
        in0_mcast_sender_noc_y.push_back((std::uint32_t)device->worker_core_from_logical_core(core).y);
    }

    // in0 sender: mcast sender runtime args
    uint32_t sender_id = 0;
    for (auto core : mcast_senders_coords) {
        std::vector<uint32_t> mm_in0_sender_args;

        // mcast sender - 1, mcast sender + compute core - 2
        uint32_t worker_core_type;
        if (std::find(storage_worker_common.begin(), storage_worker_common.end(), core) !=
            storage_worker_common.end()) {
            worker_core_type = 2;
        } else {
            worker_core_type = 1;
        }

        mm_in0_sender_args.push_back((std::uint32_t)worker_core_type);
        mm_in0_sender_args.push_back((std::uint32_t)sender_id);
        mm_in0_sender_args.push_back(
            (std::uint32_t)((core == input_all_storage_cores_vec.back()) and (in0_last_ktile_w > 0)));
        mm_in0_sender_args.insert(
            mm_in0_sender_args.end(), in0_mcast_sender_noc_x.begin(), in0_mcast_sender_noc_x.end());
        mm_in0_sender_args.insert(
            mm_in0_sender_args.end(), in0_mcast_sender_noc_y.begin(), in0_mcast_sender_noc_y.end());

        in0_sender_desc.runtime_args.emplace_back(core, std::move(mm_in0_sender_args));
        sender_id++;
    }

    // in0 sender: mcast receiver runtime args
    std::vector<CoreCoord> mcast_receiver_coords = corerange_to_cores(mcast_receivers);
    for (auto core : mcast_receiver_coords) {
        std::vector<uint32_t> mm_in0_receiver_args;
        // mcast receiver - 3
        uint32_t worker_core_type = 3;
        mm_in0_receiver_args.push_back((std::uint32_t)worker_core_type);
        mm_in0_receiver_args.push_back((std::uint32_t)0);
        mm_in0_receiver_args.push_back((std::uint32_t)0);  // in0_last_ktile_w
        mm_in0_receiver_args.insert(
            mm_in0_receiver_args.end(), in0_mcast_sender_noc_x.begin(), in0_mcast_sender_noc_x.end());
        mm_in0_receiver_args.insert(
            mm_in0_receiver_args.end(), in0_mcast_sender_noc_y.begin(), in0_mcast_sender_noc_y.end());

        in0_sender_desc.runtime_args.emplace_back(core, std::move(mm_in0_receiver_args));
    }

    // in0 sender: idle core runtime args
    for (auto core : all_cores_in_rect_grid_vec) {
        if (std::find(mcast_senders_coords.begin(), mcast_senders_coords.end(), core) == mcast_senders_coords.end() &&
            std::find(mcast_receiver_coords.begin(), mcast_receiver_coords.end(), core) ==
                mcast_receiver_coords.end()) {
            // idle core - 0
            in0_sender_desc.runtime_args.emplace_back(core, KernelDescriptor::CoreRuntimeArgs{(std::uint32_t)0});
        }
    }

    // in1 sender/writer & compute: set idle/worker designation for all cores in bounding box
    for (auto core : all_cores_in_rect_grid_vec) {
        if (std::find(all_worker_cores.ranges().begin(), all_worker_cores.ranges().end(), core) ==
            all_worker_cores.ranges().end()) {
            // not a worker core
            in1_sender_writer_desc.runtime_args.emplace_back(core, KernelDescriptor::CoreRuntimeArgs{(std::uint32_t)0});
            compute_desc.runtime_args.emplace_back(core, KernelDescriptor::CoreRuntimeArgs{(std::uint32_t)0});
        } else {
            // worker core - compute gets is_worker_core=1
            compute_desc.runtime_args.emplace_back(core, KernelDescriptor::CoreRuntimeArgs{(std::uint32_t)1});
        }
    }

    // Build output NOC coordinates
    std::vector<uint32_t> output_noc_x;
    std::vector<uint32_t> output_noc_y;
    std::vector<CoreCoord> output_coords = corerange_to_cores(output_all_storage_cores, std::nullopt, true);
    output_noc_x.reserve(output_coords.size());
    for (auto core : output_coords) {
        output_noc_x.push_back((std::uint32_t)device->worker_core_from_logical_core(core).x);
    }
    output_noc_y.reserve(output_coords.size());
    for (auto core : output_coords) {
        output_noc_y.push_back((std::uint32_t)device->worker_core_from_logical_core(core).y);
    }

    uint32_t num_cores_written_back = (N + per_core_N_storage - 1) / per_core_N_storage;
    uint32_t expected_max_total_width = num_cores_written_back * per_core_N_storage;
    uint32_t total_tensor_width_written_back = 0;

    // in1 sender/writer: worker core runtime args (complex resharding logic)
    uint32_t bank_id = 0;
    std::vector<uint32_t> bank_ids;
    uint32_t curr_storage_core_idx = 0;
    uint32_t per_core_N_storage_curr_stride = 0;

    uint32_t worker_core_stride = 0;
    uint32_t storage_core_stride = 0;
    uint32_t curr_worker_core = 0;
    uint32_t curr_storage_core = 0;

    for (uint32_t i = 0; i < all_worker_cores_ordered.size(); ++i) {
        auto core = all_worker_cores_ordered[i];

        bool is_worker_core = true;
        std::vector<uint32_t> mm_in1_sender_writer_args;
        mm_in1_sender_writer_args.push_back((std::uint32_t)is_worker_core);
        mm_in1_sender_writer_args.push_back(in1_buffer->address());
        if (bias_buffer != nullptr) {
            mm_in1_sender_writer_args.push_back(bias_buffer->address());
        } else {
            mm_in1_sender_writer_args.push_back(0);
        }

        uint32_t vc = bank_id & 0x3;
        bank_ids.push_back(bank_id);
        for (uint32_t j = 0; j < i; ++j) {
            auto core_prev = all_worker_cores_ordered[j];
            if (core_prev.y == core.y and ((bank_id & 0x3) == (bank_ids[j] & 0x3))) {  // same vc and same row
                vc = (vc + 1) & 0x3;
                break;
            }
        }
        mm_in1_sender_writer_args.push_back((std::uint32_t)bank_id);
        mm_in1_sender_writer_args.push_back((std::uint32_t)vc);

        bank_id = (bank_id + 1) % num_dram_banks;

        if (per_core_N_in1_sender < per_core_N_storage) {
            TT_FATAL(curr_storage_core_idx < num_cores_written_back, "Worker {} has no storage area assigned", core);

            uint32_t remaining_per_core_N_storage = (per_core_N_storage - per_core_N_storage_curr_stride);
            uint32_t per_core_N_reshard_1 = (remaining_per_core_N_storage > per_core_N_in1_sender)
                                                ? per_core_N_in1_sender
                                                : remaining_per_core_N_storage;
            uint32_t per_core_N_reshard_2 = per_core_N_in1_sender - per_core_N_reshard_1;

            if (per_core_N_reshard_2 != 0 and (curr_storage_core_idx + 1) < num_cores_written_back) {
                mm_in1_sender_writer_args.push_back(2);
            } else {
                mm_in1_sender_writer_args.push_back(1);
            }

            mm_in1_sender_writer_args.push_back(
                per_core_N_storage_curr_stride * output_single_tile_size);  // reshard_tensor_start_offset
            mm_in1_sender_writer_args.push_back(
                per_core_N_reshard_1 * output_single_tile_size);                       // per_core_N_reshard_bytes_1
            mm_in1_sender_writer_args.push_back(output_noc_x[curr_storage_core_idx]);  // output_noc_x
            mm_in1_sender_writer_args.push_back(output_noc_y[curr_storage_core_idx]);  // output_noc_y

            total_tensor_width_written_back += per_core_N_reshard_1;

            if (per_core_N_reshard_2 != 0 and (curr_storage_core_idx + 1) < num_cores_written_back) {
                mm_in1_sender_writer_args.push_back(
                    per_core_N_reshard_2 * output_single_tile_size);  // per_core_N_reshard_bytes_2
                mm_in1_sender_writer_args.push_back(output_noc_x[curr_storage_core_idx + 1]);  // output_noc_x
                mm_in1_sender_writer_args.push_back(output_noc_y[curr_storage_core_idx + 1]);  // output_noc_y

                total_tensor_width_written_back += per_core_N_reshard_2;
            }

            curr_storage_core_idx += (per_core_N_storage_curr_stride + per_core_N_in1_sender) / per_core_N_storage;
            per_core_N_storage_curr_stride =
                (per_core_N_storage_curr_stride + per_core_N_in1_sender) % per_core_N_storage;
        } else {
            uint32_t num_cores_write_back = 0;

            if (curr_storage_core < num_cores_written_back) {
                num_cores_write_back++;

                worker_core_stride = per_core_N_storage - storage_core_stride;

                mm_in1_sender_writer_args.push_back(
                    storage_core_stride * output_single_tile_size);  // reshard_tensor_start_offset
                mm_in1_sender_writer_args.push_back(
                    worker_core_stride * output_single_tile_size);                     // per_core_N_reshard
                mm_in1_sender_writer_args.push_back(output_noc_x[curr_storage_core]);  // output_noc_x
                mm_in1_sender_writer_args.push_back(output_noc_y[curr_storage_core]);  // output_noc_y

                curr_storage_core += (storage_core_stride + worker_core_stride) / per_core_N_storage;
                storage_core_stride = (storage_core_stride + worker_core_stride) % per_core_N_storage;

                if (worker_core_stride >= per_core_N_in1_sender) {
                    curr_worker_core += 1;
                }

                total_tensor_width_written_back += worker_core_stride;

                while (curr_worker_core <= i and curr_storage_core < num_cores_written_back) {
                    num_cores_write_back++;

                    bool increment_worker_core = (worker_core_stride + per_core_N_storage) >= per_core_N_in1_sender;
                    uint32_t current_worker_stride_total =
                        increment_worker_core ? per_core_N_in1_sender : worker_core_stride + per_core_N_storage;
                    uint32_t current_worker_write_back_tiles = current_worker_stride_total - worker_core_stride;

                    if (increment_worker_core) {
                        curr_worker_core += 1;
                    }

                    mm_in1_sender_writer_args.push_back(
                        current_worker_write_back_tiles * output_single_tile_size);        // per_core_N_reshard
                    mm_in1_sender_writer_args.push_back(output_noc_x[curr_storage_core]);  // output_noc_x
                    mm_in1_sender_writer_args.push_back(output_noc_y[curr_storage_core]);  // output_noc_y

                    total_tensor_width_written_back += current_worker_write_back_tiles;

                    storage_core_stride = current_worker_write_back_tiles % per_core_N_storage;
                    curr_storage_core += current_worker_write_back_tiles / per_core_N_storage;
                    worker_core_stride = current_worker_stride_total;
                }
            }

            mm_in1_sender_writer_args.insert(mm_in1_sender_writer_args.begin() + 5, num_cores_write_back);
        }

        TT_FATAL(
            mm_in1_sender_writer_args.size() >= 10,
            "Kernel requires at least 10 runtime args, got {}",
            mm_in1_sender_writer_args.size());

        in1_sender_writer_desc.runtime_args.emplace_back(core, std::move(mm_in1_sender_writer_args));
    }

    TT_FATAL(
        total_tensor_width_written_back <= expected_max_total_width,
        "more datums written back to sharded tensor, L1 corruption, expected: {}, actual: {}",
        expected_max_total_width,
        total_tensor_width_written_back);

    // Push kernels to descriptor
    desc.kernels.push_back(std::move(in0_sender_desc));
    desc.kernels.push_back(std::move(in1_sender_writer_desc));
    desc.kernels.push_back(std::move(compute_desc));

    return desc;
}

}  // namespace ttnn::prim::matmul_new_detail

// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/quasar/matmul/device/factory/matmul_multicore_reuse_mcast_dram_sharded_program_factory.hpp"
#include "ttnn/operations/experimental/quasar/matmul/device/utilities/matmul_utilities.hpp"
#include "ttnn/operations/experimental/quasar/matmul/device/config/matmul_program_config.hpp"

#include <algorithm>
#include <utility>

#include "hostdevcommon/common_values.hpp"
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/tt_align.hpp>
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "ttnn/operations/compute_throttle_utils.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/operations/experimental/quasar/matmul/shared_with_host/activation_type.hpp"

using namespace tt;

using ttnn::operations::unary::UnaryOpType;
using ttnn::operations::unary::UnaryWithParam;

using tt::tt_metal::CBDescriptor;
using tt::tt_metal::CBFormatDescriptor;
using tt::tt_metal::ComputeConfigDescriptor;
using tt::tt_metal::DataMovementConfigDescriptor;
using tt::tt_metal::KernelDescriptor;
using tt::tt_metal::ProgramDescriptor;

namespace ttnn::prim::qsr {
namespace reuse_dram_sharded_optimized_helpers {

using dram_sharded_helpers::get_max_page_size_and_num_pages;
using dram_sharded_helpers::get_optimal_dram_bank_to_reader_assignment;
using dram_sharded_helpers::move_common_entries;

static ProgramDescriptor create_program_dram_sharded_descriptor(
    tt::tt_metal::IDevice* device,
    const CoreRangeSet& input_all_storage_cores,
    const CoreRangeSet& output_all_storage_cores,
    MathFidelity math_fidelity,
    bool fp32_dest_acc_en,
    bool math_approx_mode,
    bool packer_l1_acc,
    bool dst_full_sync_en,
    ttnn::operations::compute_throttle_utils::ThrottleLevel throttle_level,
    uint32_t B,
    uint32_t /* M */,
    uint32_t N,
    uint32_t K,
    uint32_t in0_block_w,
    uint32_t in0_last_ktile_w,
    uint32_t per_core_M,
    uint32_t per_core_N_storage,
    std::optional<UnaryWithParam> fused_activation,
    const tt::tt_metal::MeshTensor& in0_tensor,
    const tt::tt_metal::MeshTensor& in1_tensor,
    ttsl::optional_reference<const tt::tt_metal::MeshTensor> bias_tensor,
    const tt::tt_metal::MeshTensor& out_tensor,
    const tt::tt_metal::Tile& in0_tile,
    const tt::tt_metal::Tile& in1_tile,
    const tt::tt_metal::Tile& bias_tile,
    const tt::tt_metal::Tile& output_tile,
    tt::DataFormat in0_data_format,
    tt::DataFormat in1_data_format,
    tt::DataFormat bias_data_format,
    tt::DataFormat output_data_format,
    bool untilize_out,
    bool skip_compute,
    bool skip_in0_mcast,
    bool skip_write_back,
    bool row_broadcast_bias) {
    using namespace tt;

    ttsl::optional_reference<const tt::tt_metal::MeshTensor> bias;
    if (bias_tensor.has_value()) {
        bias = *bias_tensor;
    }

    // currently only support transpose of the full tile
    bool in1_transpose_tile = in1_tile.get_transpose_of_faces() && in1_tile.get_transpose_within_face();
    TT_FATAL(
        in1_tensor.shard_spec()->orientation == ShardOrientation::ROW_MAJOR, "Only ROW_MAJOR sharding is supported");

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

    // in1 is the reader of weights/output writer, and we choose to make it use the optimized reader noc
    tt_metal::NOC in0_noc = tt::tt_metal::detail::preferred_noc_for_dram_write(device->arch());
    tt_metal::NOC in1_noc = tt::tt_metal::detail::preferred_noc_for_dram_read(device->arch());

    CoreCoord start_core_noc = top_left_core_physical;
    CoreCoord end_core_noc = bottom_right_core_physical;
    if (in0_noc == tt::tt_metal::NOC::NOC_1) {
        std::swap(start_core_noc, end_core_noc);
    }

    // get the dram readers
    std::vector<CoreCoord> all_worker_cores_ordered;
    CoreRangeSet all_worker_cores;
    get_optimal_dram_bank_to_reader_assignment(device, all_worker_cores_ordered, all_worker_cores, in1_noc);

    // dram banks
    uint32_t num_dram_banks = all_worker_cores_ordered.size();

    // Remove cores assigned to padding-only DRAM banks from the workers category
    uint32_t in1_shard_width_tiles = in1_tensor.shard_spec()->shape[1] / in1_tile.get_tile_shape()[1];
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

    auto subblock_hw = operations::experimental::quasar::matmul::bmm_op_utils_qsr::get_matmul_subblock_params(
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

    // Number of in1 columns in the last subblock that are actually backed by reader-pushed tiles.
    // When the subblock-width optimization above pads per_core_N_compute beyond per_core_N_in1_sender,
    // the last subblock has out_subblock_w lanes total but only this many lanes correspond to in1
    // tiles the reader pushed into cb_in1; the rest are padded columns that the output writer drops.
    // The compute kernel uses this to narrow the matmul_block call for the last subblock so it never
    // reads cb_in1 tile indices that were not produced for the current block.
    // When no padding occurs (per_core_N_compute == per_core_N_in1_sender) this equals out_subblock_w.
    uint32_t last_subblock_w_valid = out_subblock_w - (per_core_N_compute - per_core_N_in1_sender);

    uint32_t num_blocks = K / in0_block_w;
    bool packer_l1_acc_en = packer_l1_acc && num_blocks > 1;

    tt::DataFormat interm0_data_format = packer_l1_acc_en
                                             ? (fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b)
                                             : (fp32_dest_acc_en ? tt::DataFormat::Float32 : output_data_format);

    uint32_t in0_single_tile_size = in0_tile.get_tile_size(in0_data_format);
    uint32_t in1_single_tile_size = in1_tile.get_tile_size(in1_data_format);
    uint32_t bias_single_tile_size = bias_tile.get_tile_size(bias_data_format);
    uint32_t output_single_tile_size = output_tile.get_tile_size(output_data_format);
    uint32_t interm0_single_tile_size = output_tile.get_tile_size(interm0_data_format);

    // in1/bias are DRAM sharded with one tile per page; the allocator pads each page to the DRAM
    // alignment (e.g. bfp8 32x16 tile = 544B padded to 576B on Blackhole's 64B alignment). The
    // reader copies blocks contiguously from DRAM, so the CB must hold tiles at the padded stride.
    const uint32_t dram_alignment = tt::tt_metal::hal::get_dram_alignment();
    uint32_t in1_aligned_tile_size = tt::align(in1_single_tile_size, dram_alignment);
    uint32_t bias_aligned_tile_size = tt::align(bias_single_tile_size, dram_alignment);

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
    uint32_t in1_CB_size = in1_CB_tiles * in1_aligned_tile_size;

    uint32_t out_block_tiles = per_core_M * per_core_N_compute;
    uint32_t out_CB_tiles = out_block_tiles;
    uint32_t out_CB_size = out_CB_tiles * output_single_tile_size;
    uint32_t interm0_CB_size = out_CB_tiles * interm0_single_tile_size;

    uint32_t out_reshard_block_tiles = per_core_M * per_core_N_storage;
    uint32_t out_reshard_CB_tiles = out_reshard_block_tiles;
    uint32_t out_reshard_CB_size = out_reshard_CB_tiles * output_single_tile_size;

    uint32_t in0_shard_width_in_tiles = in0_tensor.shard_spec()->shape[1] / in0_tile.get_tile_shape()[1];
    uint32_t in2_block_tiles = per_core_M * in0_shard_width_in_tiles;
    uint32_t in2_CB_tiles = in2_block_tiles;
    uint32_t in2_CB_size = in2_CB_tiles * in0_single_tile_size;

    uint32_t in3_CB_tiles = per_core_N_compute;
    uint32_t in3_CB_size = in3_CB_tiles * bias_aligned_tile_size;

    // get the max page size based on num tiles
    uint32_t in1_buffer_page_size, in1_buffer_num_pages;
    get_max_page_size_and_num_pages(
        device, in1_block_tiles, in1_aligned_tile_size, in1_buffer_page_size, in1_buffer_num_pages);

    uint32_t bias_buffer_page_size, bias_buffer_num_pages;
    get_max_page_size_and_num_pages(
        device, per_core_N_in1_sender, bias_aligned_tile_size, bias_buffer_page_size, bias_buffer_num_pages);

    uint32_t num_worker_cores = num_dram_banks;

    // move conflict coord from mcast receiver to mcast sender
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

    // all cores
    std::set<CoreRange> all_cores_set;
    all_cores_set.insert(mcast_senders.ranges().begin(), mcast_senders.ranges().end());
    all_cores_set.insert(mcast_receivers.ranges().begin(), mcast_receivers.ranges().end());
    CoreRangeSet all_cores = CoreRangeSet(all_cores_set);

    // grid bounding box
    CoreRange bounding_box = all_cores.bounding_box();
    std::set<CoreRange> bounding_box_set;
    bounding_box_set.insert(bounding_box);
    CoreRangeSet all_cores_in_rect_grid(bounding_box_set);
    std::vector<CoreCoord> all_cores_in_rect_grid_vec = corerange_to_cores(all_cores_in_rect_grid);

    // Semaphore IDs (manually assigned, matching CreateSemaphore sequential allocation)
    uint32_t in0_mcast_sender_semaphore_id = 0;
    uint32_t in0_mcast_receiver_semaphore_id = 1;
    uint32_t in0_mcast_sender_valid_semaphore_id = 2;

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
        (std::uint32_t)0,                                           // in0_last_ktile_h (transpose not supported)
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
        (std::uint32_t)num_blocks_per_shard,
        (std::uint32_t)in0_block_w};

    std::vector<uint32_t> in1_sender_writer_compile_time_args = {
        (std::uint32_t)in1_buffer_page_size,
        (std::uint32_t)in1_buffer_num_pages,
        // in1 block args
        (std::uint32_t)per_core_N_compute,                   // in1_block_w (padded, used only for bias CB)
        (std::uint32_t)per_core_N_in1_sender * in0_block_w,  // in1_block_num_tiles
        // in0/in1 common args
        (std::uint32_t)num_blocks,                                    // num_blocks
        (std::uint32_t)out_block_tiles,                               // out_block_num_tiles
        (std::uint32_t)per_core_N_compute * output_single_tile_size,  // out_tensor_stride_w_bytes
        (std::uint32_t)per_core_N_storage * output_single_tile_size,  // out_reshard_tensor_stride_w_bytes
        (std::uint32_t)per_core_M};
    if (bias.has_value()) {
        in1_sender_writer_compile_time_args.push_back(bias_buffer_page_size);
        in1_sender_writer_compile_time_args.push_back(bias_buffer_num_pages);
        in1_sender_writer_compile_time_args.push_back((std::uint32_t)1);
    }

    std::map<std::string, std::string> mm_kernel_defines;
    std::map<std::string, std::string> mm_kernel_in0_sender_define;
    std::map<std::string, std::string> mm_kernel_in1_sender_writer_defines;
    if (bias.has_value()) {
        mm_kernel_defines["FUSE_BIAS"] = "1";
        mm_kernel_in1_sender_writer_defines["FUSE_BIAS"] = "1";
    }
    if (fused_activation.has_value()) {
        if (fused_activation.value().op_type == UnaryOpType::RELU) {
            mm_kernel_defines["PACK_RELU"] = "1";
        } else {
            mm_kernel_defines["SFPU_ACTIVATION"] = "1";
        }
    }
    if (packer_l1_acc_en) {
        mm_kernel_defines["PACKER_L1_ACC"] = "1";
    }
    if (fp32_dest_acc_en) {
        mm_kernel_defines["FP32_DEST_ACC_EN"] = "1";
    }
    mm_kernel_in1_sender_writer_defines["OUT_SHARDED"] = "1";
    mm_kernel_in1_sender_writer_defines["SKIP_MCAST"] = "1";

    if (skip_compute) {
        mm_kernel_defines["SKIP_COMPUTE"] = "1";
    }
    if (skip_in0_mcast) {
        mm_kernel_in0_sender_define["SKIP_MCAST"] = "1";
    }
    if (skip_write_back) {
        mm_kernel_in1_sender_writer_defines["SKIP_WRITE_BACK"] = "1";
    }
    mm_kernel_defines["MATMUL_DRAM_SHARDED"] = "1";
    if (in1_transpose_tile) {
        mm_kernel_defines["IN1_TRANSPOSE_TILE"] = "1";
    }

    const uint32_t num_compute_cores = all_cores_in_rect_grid.num_cores();
    ttnn::operations::compute_throttle_utils::add_stagger_defines_if_needed(
        device->arch(), num_compute_cores, mm_kernel_defines);
    ttnn::operations::compute_throttle_utils::throttle_mm_perf(
        device->arch(), num_compute_cores, mm_kernel_defines, throttle_level);

    // Helper to convert std::map defines to KernelDescriptor::Defines (vector of pairs)
    auto map_to_defines = [](const std::map<std::string, std::string>& m) -> KernelDescriptor::Defines {
        KernelDescriptor::Defines result;
        result.reserve(m.size());
        for (const auto& [k, v] : m) {
            result.emplace_back(k, v);
        }
        return result;
    };

    ////////////////////////////////////////////////////////////////////////////
    //                      Build Kernel Descriptors
    ////////////////////////////////////////////////////////////////////////////

    // in0 sender kernel (reader - RISCV_1)
    KernelDescriptor in0_sender_kernel_desc;
    in0_sender_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/quasar/matmul/device/kernels/dataflow/"
        "reader_bmm_tile_layout_in0_sender_dram_sharded.cpp";
    in0_sender_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    in0_sender_kernel_desc.core_ranges = all_cores_in_rect_grid;
    in0_sender_kernel_desc.compile_time_args = in0_sender_compile_time_args;
    in0_sender_kernel_desc.defines = map_to_defines(mm_kernel_in0_sender_define);
    in0_sender_kernel_desc.named_compile_time_args = {
        {"cb_in0", tt::CBIndex::c_0},
        {"cb_in0_sharded", tt::CBIndex::c_2},
    };
    in0_sender_kernel_desc.config =
        DataMovementConfigDescriptor{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = in0_noc};

    // in1 sender/writer kernel (writer - RISCV_0)
    KernelDescriptor in1_sender_writer_kernel_desc;
    in1_sender_writer_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/quasar/matmul/device/kernels/dataflow/"
        "reader_bmm_tile_layout_in1_sender_dram_sharded.cpp";
    in1_sender_writer_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    in1_sender_writer_kernel_desc.core_ranges = all_cores_in_rect_grid;
    in1_sender_writer_kernel_desc.compile_time_args = in1_sender_writer_compile_time_args;
    in1_sender_writer_kernel_desc.defines = map_to_defines(mm_kernel_in1_sender_writer_defines);
    in1_sender_writer_kernel_desc.named_compile_time_args = {
        {"cb_in1", tt::CBIndex::c_1},
        {"cb_bias", tt::CBIndex::c_3},
        {"cb_out", tt::CBIndex::c_4},
        {"cb_out_reshard", tt::CBIndex::c_6},
    };
    in1_sender_writer_kernel_desc.config =
        DataMovementConfigDescriptor{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = in1_noc};

    // Compute kernel
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

        untilize_out,  // untilize_out
        false,         // get_batch_from_reader
        false,         // in0_transpose_tile
    };
    if (bias.has_value()) {
        compute_kernel_args.push_back(row_broadcast_bias ? 1u : 0u);
    }

    KernelDescriptor compute_kernel_desc;
    compute_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/quasar/matmul/device/kernels/compute/"
        "bmm_large_block_zm_fused_bias_activation.cpp";
    compute_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_kernel_desc.core_ranges = all_cores_in_rect_grid;
    compute_kernel_desc.compile_time_args = compute_kernel_args;
    compute_kernel_desc.defines = map_to_defines(mm_kernel_defines);
    {
        KernelDescriptor::NamedCompileTimeArgs named_compile_args = {
            {"cb_in0", tt::CBIndex::c_0},
            {"cb_in1", tt::CBIndex::c_1},
            {"cb_bias", tt::CBIndex::c_3},
            {"cb_out", tt::CBIndex::c_4},
            {"cb_intermed0", tt::CBIndex::c_5},
            {"cb_in0_intermediate", tt::CBIndex::c_8},
            {"cb_in1_intermediate", tt::CBIndex::c_9},
            {"cb_in0_transposed", tt::CBIndex::c_10},
            {"bias_ntiles", per_core_N_compute},
            {"last_subblock_w_valid", last_subblock_w_valid},
        };
        if (fused_activation.has_value() && fused_activation.value().op_type != UnaryOpType::RELU) {
            using ttnn::operations::experimental::quasar::matmul::utilities::get_activation_params;
            const auto params = get_activation_params(fused_activation.value());
            named_compile_args.push_back({"activation_type", static_cast<uint32_t>(params.type)});
            named_compile_args.push_back({"activation_param0", params.param0});
            named_compile_args.push_back({"activation_param1", params.param1});
            named_compile_args.push_back({"activation_param2", params.param2});
        }
        compute_kernel_desc.named_compile_time_args = std::move(named_compile_args);
    }
    compute_kernel_desc.config = ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .dst_full_sync_en = dst_full_sync_en,
        .math_approx_mode = math_approx_mode,
    };

    ////////////////////////////////////////////////////////////////////////////
    //                      Build CBDescriptors
    ////////////////////////////////////////////////////////////////////////////
    ProgramDescriptor desc;

    tt::tt_metal::TileDescriptor in0_tile_desc{in0_tile};
    tt::tt_metal::TileDescriptor in1_tile_desc{in1_tile};
    tt::tt_metal::TileDescriptor bias_tile_desc{bias_tile};
    tt::tt_metal::TileDescriptor output_tile_desc{output_tile};

    // CB 0: in0
    {
        CBDescriptor cb_desc;
        cb_desc.total_size = in0_CB_size;
        cb_desc.core_ranges = all_cores_in_rect_grid;
        cb_desc.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_0,
            .data_format = in0_data_format,
            .page_size = in0_single_tile_size,
            .tile = in0_tile_desc});
        desc.cbs.push_back(std::move(cb_desc));
    }

    // CB 1: in1
    {
        CBDescriptor cb_desc;
        cb_desc.total_size = in1_CB_size;
        cb_desc.core_ranges = all_cores_in_rect_grid;
        cb_desc.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_1,
            .data_format = in1_data_format,
            .page_size = in1_aligned_tile_size,
            .tile = in1_tile_desc});
        desc.cbs.push_back(std::move(cb_desc));
    }

    // CB 2: in0 sharded (backed by in0_buffer)
    {
        CBDescriptor cb_desc;
        cb_desc.total_size = in2_CB_size;
        cb_desc.core_ranges = all_cores_in_rect_grid;
        cb_desc.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_2,
            .data_format = in0_data_format,
            .page_size = in0_single_tile_size,
            .tile = in0_tile_desc});
        cb_desc.tensor = &in0_tensor;
        desc.cbs.push_back(std::move(cb_desc));
    }

    // CB 4 and CB 5: output and intermediate
    if ((interm0_data_format != output_data_format) || (untilize_out && (in1_num_subblocks > 1))) {
        // Separate output and intermediate CBs
        {
            CBDescriptor cb_desc;
            cb_desc.total_size = out_CB_size;
            cb_desc.core_ranges = all_cores_in_rect_grid;
            cb_desc.format_descriptors.push_back(CBFormatDescriptor{
                .buffer_index = tt::CBIndex::c_4,
                .data_format = output_data_format,
                .page_size = output_single_tile_size,
                .tile = output_tile_desc});
            desc.cbs.push_back(std::move(cb_desc));
        }
        {
            CBDescriptor cb_desc;
            cb_desc.total_size = interm0_CB_size;
            cb_desc.core_ranges = all_cores_in_rect_grid;
            cb_desc.format_descriptors.push_back(CBFormatDescriptor{
                .buffer_index = tt::CBIndex::c_5,
                .data_format = interm0_data_format,
                .page_size = interm0_single_tile_size,
                .tile = output_tile_desc});
            desc.cbs.push_back(std::move(cb_desc));
        }
    } else {
        // share buffer
        CBDescriptor cb_desc;
        cb_desc.total_size = out_CB_size;
        cb_desc.core_ranges = all_cores_in_rect_grid;
        cb_desc.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_4,
            .data_format = output_data_format,
            .page_size = output_single_tile_size,
            .tile = output_tile_desc});
        cb_desc.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_5,
            .data_format = interm0_data_format,
            .page_size = interm0_single_tile_size,
            .tile = output_tile_desc});
        desc.cbs.push_back(std::move(cb_desc));
    }

    // CB 6: resharded output (backed by out_buffer)
    {
        CBDescriptor cb_desc;
        cb_desc.total_size = out_reshard_CB_size;
        cb_desc.core_ranges = all_cores_in_rect_grid;
        cb_desc.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_6,
            .data_format = output_data_format,
            .page_size = output_single_tile_size,
            .tile = output_tile_desc});
        cb_desc.tensor = &out_tensor;
        desc.cbs.push_back(std::move(cb_desc));
    }

    // CB 3: bias
    if (bias.has_value()) {
        CBDescriptor cb_desc;
        cb_desc.total_size = in3_CB_size;
        cb_desc.core_ranges = all_cores_in_rect_grid;
        cb_desc.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_3,
            .data_format = bias_data_format,
            .page_size = bias_aligned_tile_size,
            .tile = bias_tile_desc});
        desc.cbs.push_back(std::move(cb_desc));
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Semaphore Descriptors
    ////////////////////////////////////////////////////////////////////////////
    desc.semaphores.push_back(tt::tt_metal::SemaphoreDescriptor{
        .id = in0_mcast_sender_semaphore_id, .core_ranges = all_cores_in_rect_grid, .initial_value = INVALID});
    desc.semaphores.push_back(tt::tt_metal::SemaphoreDescriptor{
        .id = in0_mcast_receiver_semaphore_id, .core_ranges = all_cores_in_rect_grid, .initial_value = INVALID});
    desc.semaphores.push_back(tt::tt_metal::SemaphoreDescriptor{
        .id = in0_mcast_sender_valid_semaphore_id, .core_ranges = all_cores_in_rect_grid, .initial_value = VALID});

    ////////////////////////////////////////////////////////////////////////////
    //                      Runtime Args (per-core loop)
    ////////////////////////////////////////////////////////////////////////////

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

    // in0 sender runtime args (mcast senders)
    uint32_t sender_id = 0;
    for (auto core : mcast_senders_coords) {
        std::vector<uint32_t> mm_in0_sender_args;

        uint32_t worker_core_type;
        if (find(storage_worker_common.begin(), storage_worker_common.end(), core) != storage_worker_common.end()) {
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

        in0_sender_kernel_desc.runtime_args.emplace_back(core, mm_in0_sender_args);
        sender_id++;
    }

    // in0 sender runtime args (mcast receivers)
    std::vector<CoreCoord> mcast_receiver_coords = corerange_to_cores(mcast_receivers);
    for (auto core : mcast_receiver_coords) {
        std::vector<uint32_t> mm_in0_receiver_args;
        uint32_t worker_core_type = 3;
        mm_in0_receiver_args.push_back((std::uint32_t)worker_core_type);
        mm_in0_receiver_args.push_back((std::uint32_t)0);
        mm_in0_receiver_args.push_back((std::uint32_t)0);  // in0_last_ktile_w
        mm_in0_receiver_args.insert(
            mm_in0_receiver_args.end(), in0_mcast_sender_noc_x.begin(), in0_mcast_sender_noc_x.end());
        mm_in0_receiver_args.insert(
            mm_in0_receiver_args.end(), in0_mcast_sender_noc_y.begin(), in0_mcast_sender_noc_y.end());

        in0_sender_kernel_desc.runtime_args.emplace_back(core, mm_in0_receiver_args);
    }

    // in0 sender runtime args (idle cores in rect grid)
    for (auto core : all_cores_in_rect_grid_vec) {
        if (std::find(mcast_senders_coords.begin(), mcast_senders_coords.end(), core) == mcast_senders_coords.end() and
            std::find(mcast_receiver_coords.begin(), mcast_receiver_coords.end(), core) ==
                mcast_receiver_coords.end()) {
            std::vector<uint32_t> mm_in0_idle_args;
            uint32_t worker_core_type = 0;
            mm_in0_idle_args.push_back((std::uint32_t)worker_core_type);

            in0_sender_kernel_desc.runtime_args.emplace_back(core, mm_in0_idle_args);
        }
    }

    // Compute and in1 sender/writer runtime args
    uint32_t bank_id = 0;
    std::vector<uint32_t> bank_ids;
    uint32_t curr_storage_core_idx = 0;
    uint32_t per_core_N_storage_curr_stride = 0;

    uint32_t worker_core_stride = 0;
    uint32_t storage_core_stride = 0;
    uint32_t curr_worker_core = 0;
    uint32_t curr_storage_core = 0;

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

    // For all cores in the rect grid, set compute and in1 rt args for non-worker cores
    for (auto core : all_cores_in_rect_grid_vec) {
        if (std::find(all_worker_cores.ranges().begin(), all_worker_cores.ranges().end(), core) ==
            all_worker_cores.ranges().end()) {  // not worker
            bool is_worker_core = false;
            std::vector<uint32_t> mm_in1_sender_writer_args;
            mm_in1_sender_writer_args.push_back((std::uint32_t)is_worker_core);
            in1_sender_writer_kernel_desc.runtime_args.emplace_back(core, mm_in1_sender_writer_args);

            std::vector<uint32_t> mm_compute_args;
            mm_compute_args.push_back((std::uint32_t)is_worker_core);
            compute_kernel_desc.runtime_args.emplace_back(core, mm_compute_args);
        } else {
            bool is_worker_core = true;
            std::vector<uint32_t> mm_compute_args;
            mm_compute_args.push_back((std::uint32_t)is_worker_core);
            compute_kernel_desc.runtime_args.emplace_back(core, mm_compute_args);
        }
    }

    // Worker cores: in1 sender/writer runtime args
    for (uint32_t i = 0; i < all_worker_cores_ordered.size(); ++i) {
        auto core = all_worker_cores_ordered[i];

        bool is_worker_core = true;
        std::vector<uint32_t> mm_in1_sender_writer_args;
        mm_in1_sender_writer_args.push_back((std::uint32_t)is_worker_core);
        mm_in1_sender_writer_args.push_back(in1_tensor.address());  // [1]: will be replaced by Buffer*
        mm_in1_sender_writer_args.push_back(bias.has_value() ? bias->address() : 0u);  // [2]: may be replaced

        uint32_t vc = bank_id & 0x3;
        bank_ids.push_back(bank_id);
        for (uint32_t j = 0; j < i; ++j) {
            auto core_prev = all_worker_cores_ordered[j];

            if (core_prev.y == core.y and ((bank_id & 0x3) == (bank_ids[j] & 0x3))) {
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

        // Build variant args: positions [1] and [2] are buffer addresses
        std::vector<std::variant<uint32_t, std::reference_wrapper<const tt::tt_metal::MeshTensor>>> in1_writer_args(
            mm_in1_sender_writer_args.begin(), mm_in1_sender_writer_args.end());
        in1_writer_args[1] = in1_tensor;
        if (bias.has_value()) {
            in1_writer_args[2] = *bias;
        }
        in1_sender_writer_kernel_desc.emplace_runtime_args(core, in1_writer_args);
        TT_FATAL(
            mm_in1_sender_writer_args.size() >= 10,
            "Kernel requires at least 10 runtime args, got {}",
            mm_in1_sender_writer_args.size());
    }

    TT_FATAL(
        total_tensor_width_written_back <= expected_max_total_width,
        "more datums written back to sharded tensor, L1 corruption, expected: {}, actual: {}",
        expected_max_total_width,
        total_tensor_width_written_back);

    ////////////////////////////////////////////////////////////////////////////
    //                      Push Kernel Descriptors
    ////////////////////////////////////////////////////////////////////////////
    desc.kernels.push_back(std::move(in0_sender_kernel_desc));
    desc.kernels.push_back(std::move(in1_sender_writer_kernel_desc));
    desc.kernels.push_back(std::move(compute_kernel_desc));

    return desc;
}

}  // namespace reuse_dram_sharded_optimized_helpers

ProgramDescriptor MatmulMultiCoreReuseMultiCastDRAMShardedProgramFactory::create_descriptor(
    const ttnn::prim::qsr::MatmulParams& operation_attributes,
    const ttnn::prim::qsr::MatmulInputs& tensor_args,
    std::vector<ttnn::Tensor>& tensor_return_value,
    const std::optional<CoreRangeSet>& /*core_range_set*/) {
    const auto& input_tensors = tensor_args.input_tensors;
    const auto& optional_input_tensors = tensor_args.optional_input_tensors;
    const auto& output_tensors = tensor_return_value;

    const auto& a = input_tensors.at(0);
    const auto& b = input_tensors.at(1).mesh_tensor();
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
    tt::DataFormat in0_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());
    tt::DataFormat in1_data_format = tt::tt_metal::datatype_to_dataformat_converter(b.dtype());
    tt::DataFormat output_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());

    ttsl::optional_reference<const tt::tt_metal::MeshTensor> bias_mesh;
    tt::DataFormat bias_data_format = tt::DataFormat::Bfp8_b;
    if (bias.has_value()) {
        const auto& c = bias.value();
        TT_FATAL(
            c.storage_type() == StorageType::DEVICE,
            "Bias tensor must be on device, got storage type: {}",
            c.storage_type());
        TT_FATAL(a.device() == c.device(), "Operands to matmul need to be on the same device!");

        bias_mesh = c.mesh_tensor();
        bias_data_format = tt::tt_metal::datatype_to_dataformat_converter(c.dtype());
    }

    const bool row_broadcast_bias =
        operations::experimental::quasar::matmul::utilities::fused_matmul_bias_row_broadcastable(bias);

    tt::tt_metal::IDevice* device = a.device();

    TT_FATAL(
        a.shard_spec().has_value() && output.shard_spec().has_value(), "Both input A and output must have shard specs");
    CoreRangeSet input_all_cores_storage = a.shard_spec().value().grid;
    CoreRangeSet output_all_cores_storage = output.shard_spec().value().grid;

    uint32_t in0_single_tile_size = in0_tile.get_tile_size(in0_data_format);
    uint32_t in1_single_tile_size = in1_tile.get_tile_size(in1_data_format);
    const auto& a_mesh = a.mesh_tensor();
    TT_FATAL(
        a_mesh.mesh_buffer().device_local_size() % in0_single_tile_size == 0,
        "Input A buffer size ({}) must be divisible by single tile size ({})",
        a_mesh.mesh_buffer().device_local_size(),
        in0_single_tile_size);
    TT_FATAL(
        b.mesh_buffer().device_local_size() % in1_single_tile_size == 0,
        "Input B buffer size ({}) must be divisible by single tile size ({})",
        b.mesh_buffer().device_local_size(),
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
    const auto& program_config =
        std::get<operations::experimental::quasar::matmul::MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig>(
            operation_attributes.program_config.value());
    const auto& in0_block_w = program_config.in0_block_w;
    const auto& per_core_M = program_config.per_core_M;
    const auto& per_core_N = program_config.per_core_N;
    const auto& fused_activation = program_config.fused_activation;

    const auto& untilize_out = operation_attributes.untilize_out;
    const bool skip_compute = false;
    const bool skip_in0_mcast = false;
    const bool skip_write_back = false;

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);

    uint32_t B = 1;
    uint32_t Mt = get_batch_size(ashape) * ashape[-2] / in0_tile_shape[0];
    uint32_t Kt = ashape[-1] / in0_tile_shape[1];
    uint32_t Nt = bshape[-1] / in1_tile_shape[1];
    uint32_t in0_last_ktile_w = a.logical_shape()[-1] % in0_tile_shape[1];

    TT_FATAL(Kt % in0_block_w == 0, "Kt ({}) must be divisible by in0_block_w ({})", Kt, in0_block_w);

    const auto& output_mesh = output.mesh_tensor();

    return reuse_dram_sharded_optimized_helpers::create_program_dram_sharded_descriptor(
        device,
        input_all_cores_storage,
        output_all_cores_storage,
        math_fidelity,
        fp32_dest_acc_en,
        math_approx_mode,
        packer_l1_acc,
        dst_full_sync_en,
        ttnn::get_throttle_level(operation_attributes.compute_kernel_config),
        B,
        Mt,
        Nt,
        Kt,
        in0_block_w,
        in0_last_ktile_w,
        per_core_M,
        per_core_N,
        fused_activation,
        a_mesh,
        b,
        bias_mesh,
        output_mesh,
        in0_tile,
        in1_tile,
        bias.has_value() ? bias->tensor_spec().tile() : output_tile,
        output_tile,
        in0_data_format,
        in1_data_format,
        bias_data_format,
        output_data_format,
        untilize_out,
        skip_compute,
        skip_in0_mcast,
        skip_write_back,
        row_broadcast_bias);
}

}  // namespace ttnn::prim::qsr

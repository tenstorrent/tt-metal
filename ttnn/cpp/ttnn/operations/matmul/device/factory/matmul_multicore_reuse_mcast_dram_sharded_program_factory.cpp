// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_dram_sharded_program_factory.hpp"
#include "ttnn/operations/matmul/device/utilities/matmul_utilities.hpp"
#include "ttnn/operations/matmul/device/config/matmul_program_config.hpp"

#include <algorithm>
#include <utility>

#include "hostdevcommon/common_values.hpp"
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "ttnn/operations/compute_throttle_utils.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/operations/matmul/shared_with_host/activation_type.hpp"

using namespace tt;

using ttnn::operations::unary::UnaryOpType;
using ttnn::operations::unary::UnaryWithParam;

using tt::tt_metal::KernelBuildOptLevel;
using tt::tt_metal::UnpackMode;
using tt::tt_metal::experimental::AddRuntimeArgsForNode;
using tt::tt_metal::experimental::AdvancedKernelRunArgs;
using tt::tt_metal::experimental::ComputeHardwareConfig;
using tt::tt_metal::experimental::DataflowBufferSpec;
using tt::tt_metal::experimental::DataMovementGen1Config;
using tt::tt_metal::experimental::DFBBinding;
using tt::tt_metal::experimental::DFBEndpointType;
using tt::tt_metal::experimental::DFBSpecName;
using tt::tt_metal::experimental::Group;
using tt::tt_metal::experimental::KernelRunArgs;
using tt::tt_metal::experimental::KernelSpec;
using tt::tt_metal::experimental::KernelSpecName;
using tt::tt_metal::experimental::ProgramRunArgs;
using tt::tt_metal::experimental::ProgramSpec;
using tt::tt_metal::experimental::SemaphoreBinding;
using tt::tt_metal::experimental::SemaphoreSpec;
using tt::tt_metal::experimental::SemaphoreSpecName;
using tt::tt_metal::experimental::Table;
using tt::tt_metal::experimental::TensorBinding;
using tt::tt_metal::experimental::TensorParameter;
using tt::tt_metal::experimental::TensorParamName;
using tt::tt_metal::experimental::unpack_modes;
using tt::tt_metal::experimental::WorkUnitSpec;

namespace ttnn::prim {
namespace reuse_dram_sharded_optimized_helpers {

using dram_sharded_helpers::get_dram_bank_reader_assignments;
using dram_sharded_helpers::get_max_page_size_and_num_pages;
using dram_sharded_helpers::move_common_entries;
using dram_sharded_helpers::validate_num_workers_per_dram_bank;

static ttnn::device_operation::ProgramArtifacts create_program_dram_sharded_spec(
    tt::tt_metal::IDevice* device,
    const CoreRangeSet& input_all_storage_cores,
    const CoreRangeSet& output_all_storage_cores,
    ComputeHardwareConfig compute_hw,
    bool fp32_dest_acc_en,
    bool packer_l1_acc,
    ttnn::operations::compute_throttle_utils::ThrottleLevel throttle_level,
    uint32_t B,
    uint32_t /* M */,
    uint32_t N,
    uint32_t K,
    uint32_t in0_block_w,
    uint32_t in0_last_ktile_w,
    uint32_t per_core_M,
    uint32_t per_core_N_storage,
    uint32_t workers_per_bank,
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

    validate_num_workers_per_dram_bank(workers_per_bank);
    TT_FATAL(
        workers_per_bank == 1 || device->arch() == tt::ARCH::BLACKHOLE,
        "Multiple workers per DRAM bank are currently supported only on Blackhole");
    TT_FATAL(
        workers_per_bank == 1 || in1_noc == tt::tt_metal::NOC::NOC_0,
        "Multiple workers per DRAM bank currently require a NOC0 data-movement kernel");

    auto reader_assignments =
        get_dram_bank_reader_assignments(device, in1_noc, workers_per_bank, input_all_storage_cores);
    uint32_t num_dram_banks = reader_assignments.size() / workers_per_bank;

    // Remove cores assigned to padding-only DRAM banks from the workers category
    uint32_t in1_shard_width_tiles = in1_tensor.shard_spec()->shape[1] / in1_tile.get_tile_shape()[1];
    uint32_t in1_tensor_padded_width_tiles = in1_shard_width_tiles * num_dram_banks;

    if (in1_tensor_padded_width_tiles > N) {
        uint32_t padding_width_tiles = in1_tensor_padded_width_tiles - N;
        uint32_t only_padding_banks = padding_width_tiles / in1_shard_width_tiles;
        TT_FATAL(
            only_padding_banks < num_dram_banks,
            "Padding banks count {} must be less than DRAM bank count {}",
            only_padding_banks,
            num_dram_banks);
        num_dram_banks -= only_padding_banks;
        reader_assignments.resize(num_dram_banks * workers_per_bank);
    }

    TT_FATAL(
        in1_shard_width_tiles % workers_per_bank == 0,
        "DRAM-sharded matmul weight shard width {} tiles per bank must be divisible by workers_per_bank {}",
        in1_shard_width_tiles,
        workers_per_bank);

    std::vector<CoreCoord> all_worker_cores_ordered;
    all_worker_cores_ordered.reserve(reader_assignments.size());
    std::set<CoreRange> worker_ranges;
    for (const auto& assignment : reader_assignments) {
        all_worker_cores_ordered.push_back(assignment.worker_core);
        worker_ranges.insert(CoreRange(assignment.worker_core));
    }
    CoreRangeSet all_worker_cores(worker_ranges);
    const uint32_t num_workers = all_worker_cores_ordered.size();

    uint32_t per_core_N_compute = div_up(N, num_workers);
    uint32_t per_core_N_in1_sender = per_core_N_compute;
    TT_FATAL(
        workers_per_bank == 1 || in1_shard_width_tiles == workers_per_bank * per_core_N_in1_sender,
        "Multi-reader DRAM-sharded matmul requires weight shard width {} tiles to equal {} workers times {} reader "
        "tiles",
        in1_shard_width_tiles,
        workers_per_bank,
        per_core_N_in1_sender);

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

    // Number of in1 columns in the last subblock that are actually backed by reader-pushed tiles.
    // When the subblock-width optimization above pads per_core_N_compute beyond per_core_N_in1_sender,
    // the last subblock has out_subblock_w lanes total but only this many lanes correspond to in1
    // tiles the reader pushed into the in1 dataflow buffer; the rest are padded columns that the
    // output writer drops. The compute kernel uses this to narrow the matmul_block call for the last
    // subblock so it never reads in1 tile indices that were not produced for the current block.
    // When no padding occurs (per_core_N_compute == per_core_N_in1_sender) this equals out_subblock_w.
    uint32_t last_subblock_w_valid = out_subblock_w - (per_core_N_compute - per_core_N_in1_sender);

    uint32_t in1_num_subblocks = (per_core_N_compute / out_subblock_w);

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
    // reader copies blocks contiguously from DRAM, so the buffer must hold tiles at the padded stride.
    const uint32_t dram_alignment = tt::tt_metal::hal::get_dram_alignment();
    uint32_t in1_aligned_tile_size = tt::align(in1_single_tile_size, dram_alignment);
    uint32_t bias_aligned_tile_size = tt::align(bias_single_tile_size, dram_alignment);

    uint32_t in0_block_tiles = per_core_M * in0_block_w;
    uint32_t in0_num_entries = in0_block_tiles;
    if (B * num_blocks > 1) {
        in0_num_entries = in0_num_entries * 2;  // double buffer
    }
    uint32_t in1_block_tiles = per_core_N_in1_sender * in0_block_w;
    uint32_t in1_num_entries = in1_block_tiles;
    if (B * num_blocks > 1) {
        in1_num_entries = in1_num_entries * 3;  // triple buffer
    }

    uint32_t out_block_tiles = per_core_M * per_core_N_compute;
    uint32_t out_num_entries = out_block_tiles;

    uint32_t out_reshard_block_tiles = per_core_M * per_core_N_storage;
    uint32_t out_reshard_num_entries = out_reshard_block_tiles;

    uint32_t in0_shard_width_in_tiles = in0_tensor.shard_spec()->shape[1] / in0_tile.get_tile_shape()[1];
    uint32_t in2_block_tiles = per_core_M * in0_shard_width_in_tiles;
    uint32_t in0_sharded_num_entries = in2_block_tiles;

    uint32_t bias_num_entries = per_core_N_compute;

    // get the max page size based on num tiles
    uint32_t in1_buffer_page_size, in1_buffer_num_pages;
    get_max_page_size_and_num_pages(
        device, in1_block_tiles, in1_aligned_tile_size, in1_buffer_page_size, in1_buffer_num_pages);

    uint32_t bias_buffer_page_size, bias_buffer_num_pages;
    get_max_page_size_and_num_pages(
        device, per_core_N_in1_sender, bias_aligned_tile_size, bias_buffer_page_size, bias_buffer_num_pages);

    uint32_t num_worker_cores = num_workers;

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

    // Spec-scope resource names. Declared function-local rather than at file scope: the matmul
    // factory .cpp files share one unity-build target, so file-scope constants with these names
    // would collide as sibling factories are ported.
    const KernelSpecName IN0_SENDER{"in0_sender"};
    const KernelSpecName IN1_SENDER_WRITER{"in1_sender_writer"};
    const KernelSpecName COMPUTE{"compute"};

    const DFBSpecName IN0_DFB{"in0"};
    const DFBSpecName IN1_DFB{"in1"};
    const DFBSpecName IN0_SHARDED_DFB{"in0_sharded"};
    const DFBSpecName BIAS_DFB{"bias"};
    const DFBSpecName OUT_DFB{"out"};
    const DFBSpecName INTERMED0_DFB{"intermed0"};
    const DFBSpecName OUT_RESHARD_DFB{"out_reshard"};

    const SemaphoreSpecName IN0_MCAST_SENDER_SEM{"in0_mcast_sender"};
    const SemaphoreSpecName IN0_MCAST_RECEIVER_SEM{"in0_mcast_receiver"};
    const SemaphoreSpecName IN0_MCAST_SENDER_VALID_SEM{"in0_mcast_sender_valid"};

    const TensorParamName IN0{"in0"};
    const TensorParamName IN1{"in1"};
    const TensorParamName BIAS{"bias"};
    const TensorParamName OUTPUT{"output"};

    std::map<std::string, std::string> mm_kernel_defines;
    std::map<std::string, std::string> mm_kernel_in0_sender_define;
    std::map<std::string, std::string> mm_kernel_in1_sender_writer_defines;
    if (bias.has_value()) {
        mm_kernel_defines["FUSE_BIAS"] = "1";
        mm_kernel_in1_sender_writer_defines["FUSE_BIAS"] = "1";
    }
    if (workers_per_bank > 1) {
        mm_kernel_in1_sender_writer_defines["SPLIT_DRAM_BANK"] = "1";
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

    // This factory never transposes in0: the legacy compute kernel took the flag as a compile-time
    // arg whose value here was a literal false. Metal 2.0 selects the in0 dataflow buffer from a
    // preprocessor condition instead, because the kernel picks its in0 handle in a parse-time
    // ternary whose both operands must resolve; leaving the define unset also leaves the
    // in0_transposed buffer unbound, which is what this factory wants.
    const bool in0_transpose_tile = false;
    if (in0_transpose_tile) {
        mm_kernel_defines["IN0_TRANSPOSE_TILE"] = "1";
    }

    const uint32_t num_compute_cores = all_cores_in_rect_grid.num_cores();
    ttnn::operations::compute_throttle_utils::add_stagger_defines_if_needed(
        device->arch(), num_compute_cores, mm_kernel_defines);
    ttnn::operations::compute_throttle_utils::throttle_mm_perf(
        device->arch(), num_compute_cores, mm_kernel_defines, throttle_level);

    ////////////////////////////////////////////////////////////////////////////
    //                      Build DataflowBufferSpecs
    ////////////////////////////////////////////////////////////////////////////

    // The output and intermediate0 buffers share one L1 region whenever the legacy factory
    // expressed them as one buffer descriptor carrying both formats. Same total size, same bound
    // kernel set, so they form a legal two-member alias clique; in the other branch they are
    // independent.
    const bool share_out_interm_buffer =
        !((interm0_data_format != output_data_format) || (untilize_out && (in1_num_subblocks > 1)));

    DataflowBufferSpec in0_dfb_spec{
        .unique_id = IN0_DFB,
        .entry_size = in0_single_tile_size,
        .num_entries = in0_num_entries,
        .data_format_metadata = in0_data_format,
        .tile_format_metadata = in0_tile,
    };
    DataflowBufferSpec in1_dfb_spec{
        .unique_id = IN1_DFB,
        .entry_size = in1_aligned_tile_size,
        .num_entries = in1_num_entries,
        .data_format_metadata = in1_data_format,
        .tile_format_metadata = in1_tile,
    };
    // in0 arrives already resident in L1 as a width shard; the buffer is a non-owning view of it.
    DataflowBufferSpec in0_sharded_dfb_spec{
        .unique_id = IN0_SHARDED_DFB,
        .entry_size = in0_single_tile_size,
        .num_entries = in0_sharded_num_entries,
        .data_format_metadata = in0_data_format,
        .tile_format_metadata = in0_tile,
        .borrowed_from = IN0,
    };
    DataflowBufferSpec out_dfb_spec{
        .unique_id = OUT_DFB,
        .entry_size = output_single_tile_size,
        .num_entries = out_num_entries,
        .data_format_metadata = output_data_format,
        .tile_format_metadata = output_tile,
    };
    DataflowBufferSpec intermed0_dfb_spec{
        .unique_id = INTERMED0_DFB,
        .entry_size = interm0_single_tile_size,
        .num_entries = out_num_entries,
        .data_format_metadata = interm0_data_format,
        .tile_format_metadata = output_tile,
    };
    // The resharded output is written straight into the output tensor's own L1 shard.
    DataflowBufferSpec out_reshard_dfb_spec{
        .unique_id = OUT_RESHARD_DFB,
        .entry_size = output_single_tile_size,
        .num_entries = out_reshard_num_entries,
        .data_format_metadata = output_data_format,
        .tile_format_metadata = output_tile,
        .borrowed_from = OUTPUT,
    };
    if (share_out_interm_buffer) {
        out_dfb_spec.advanced_options.alias_with = {INTERMED0_DFB};
        intermed0_dfb_spec.advanced_options.alias_with = {OUT_DFB};
    }

    Group<DataflowBufferSpec> dataflow_buffers;
    dataflow_buffers.reserve(bias.has_value() ? 7 : 6);
    dataflow_buffers.push_back(std::move(in0_dfb_spec));
    dataflow_buffers.push_back(std::move(in1_dfb_spec));
    dataflow_buffers.push_back(std::move(in0_sharded_dfb_spec));
    dataflow_buffers.push_back(std::move(out_dfb_spec));
    dataflow_buffers.push_back(std::move(intermed0_dfb_spec));
    dataflow_buffers.push_back(std::move(out_reshard_dfb_spec));
    if (bias.has_value()) {
        dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = BIAS_DFB,
            .entry_size = bias_aligned_tile_size,
            .num_entries = bias_num_entries,
            .data_format_metadata = bias_data_format,
            .tile_format_metadata = bias_tile,
        });
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Semaphore Specs
    ////////////////////////////////////////////////////////////////////////////

    Group<SemaphoreSpec> semaphores;
    semaphores.reserve(3);
    semaphores.push_back(SemaphoreSpec{
        .unique_id = IN0_MCAST_SENDER_SEM,
        .target_nodes = all_cores_in_rect_grid,
    });
    semaphores.push_back(SemaphoreSpec{
        .unique_id = IN0_MCAST_RECEIVER_SEM,
        .target_nodes = all_cores_in_rect_grid,
    });
    // Declared but bound by no kernel: the legacy factory allocated a third semaphore and passed
    // its id as a compile-time arg the in0 sender never reads, so nothing on device touches it.
    // Kept so the program's semaphore allocation is byte-for-byte what it was. Its initial value
    // is likewise unobservable, but carried across for the same reason.
    semaphores.push_back(SemaphoreSpec{
        .unique_id = IN0_MCAST_SENDER_VALID_SEM,
        .target_nodes = all_cores_in_rect_grid,
        .advanced_options = {.initial_value = VALID},
    });

    ////////////////////////////////////////////////////////////////////////////
    //                      Runtime Args (per-core loops)
    ////////////////////////////////////////////////////////////////////////////

    KernelRunArgs in0_run_args{.kernel = IN0_SENDER};
    KernelRunArgs in1_run_args{.kernel = IN1_SENDER_WRITER};
    KernelRunArgs compute_run_args{.kernel = COMPUTE};

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

    // The sender noc-x block followed by the sender noc-y block, exactly the layout the in0 kernel
    // indexes by block id. Same on every node, but left per-node: promoting it to a common
    // runtime arg would change dispatch semantics, which is not this port's business.
    AdvancedKernelRunArgs::Varargs in0_sender_noc_varargs;
    in0_sender_noc_varargs.reserve(in0_mcast_sender_noc_x.size() + in0_mcast_sender_noc_y.size());
    in0_sender_noc_varargs.insert(
        in0_sender_noc_varargs.end(), in0_mcast_sender_noc_x.begin(), in0_mcast_sender_noc_x.end());
    in0_sender_noc_varargs.insert(
        in0_sender_noc_varargs.end(), in0_mcast_sender_noc_y.begin(), in0_mcast_sender_noc_y.end());
    const uint32_t num_in0_sender_varargs = in0_sender_noc_varargs.size();

    // in0 sender runtime args (mcast senders)
    uint32_t sender_id = 0;
    for (auto core : mcast_senders_coords) {
        uint32_t worker_core_type;
        if (find(storage_worker_common.begin(), storage_worker_common.end(), core) != storage_worker_common.end()) {
            worker_core_type = 2;
        } else {
            worker_core_type = 1;
        }

        AddRuntimeArgsForNode(
            in0_run_args.runtime_arg_values,
            core,
            {{"worker_core_type", worker_core_type},
             {"sender_id", sender_id},
             {"is_last_ktile_padded",
              (std::uint32_t)((core == input_all_storage_cores_vec.back()) and (in0_last_ktile_w > 0))}});
        in0_run_args.advanced_options.runtime_varargs[core] = in0_sender_noc_varargs;
        sender_id++;
    }

    // in0 sender runtime args (mcast receivers)
    std::vector<CoreCoord> mcast_receiver_coords = corerange_to_cores(mcast_receivers);
    for (auto core : mcast_receiver_coords) {
        AddRuntimeArgsForNode(
            in0_run_args.runtime_arg_values,
            core,
            {{"worker_core_type", 3u}, {"sender_id", 0u}, {"is_last_ktile_padded", 0u}});
        in0_run_args.advanced_options.runtime_varargs[core] = in0_sender_noc_varargs;
    }

    // in0 sender runtime args (idle cores in rect grid). The legacy factory emitted only the
    // core-type arg here, because the kernel returns before reading anything else. Metal 2.0
    // requires every declared runtime arg on every node the kernel runs on, so the remaining
    // fields are zero-filled; the kernel still returns before reading them.
    for (auto core : all_cores_in_rect_grid_vec) {
        if (std::find(mcast_senders_coords.begin(), mcast_senders_coords.end(), core) == mcast_senders_coords.end() and
            std::find(mcast_receiver_coords.begin(), mcast_receiver_coords.end(), core) ==
                mcast_receiver_coords.end()) {
            AddRuntimeArgsForNode(
                in0_run_args.runtime_arg_values,
                core,
                {{"worker_core_type", 0u}, {"sender_id", 0u}, {"is_last_ktile_padded", 0u}});
            in0_run_args.advanced_options.runtime_varargs[core] =
                AdvancedKernelRunArgs::Varargs(num_in0_sender_varargs, 0u);
        }
    }

    // Compute and in1 sender/writer runtime args
    std::vector<uint32_t> bank_ids;
    bank_ids.reserve(all_worker_cores_ordered.size());
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

    // Non-worker in1 writer nodes carry no write-back collection at all; worker nodes carry three
    // values per shard they write back. Collected first, then padded to a single declared length.
    Table<CoreCoord, AdvancedKernelRunArgs::Varargs> in1_writer_varargs;
    uint32_t num_in1_writer_varargs = 0;

    // For all cores in the rect grid, set compute and in1 rt args for non-worker cores
    for (auto core : all_cores_in_rect_grid_vec) {
        if (std::find(all_worker_cores.ranges().begin(), all_worker_cores.ranges().end(), core) ==
            all_worker_cores.ranges().end()) {  // not worker
            bool is_worker_core = false;
            // As on the in0 sender's idle cores: the kernel returns on is_worker_core, but the
            // schema still needs every declared name present, so the rest are zero-filled.
            AddRuntimeArgsForNode(
                in1_run_args.runtime_arg_values,
                core,
                {{"is_worker_core", (std::uint32_t)is_worker_core},
                 {"dram_bank_id", 0u},
                 {"vc", 0u},
                 {"dram_reader_index", 0u},
                 {"num_shard_to_write_back", 0u},
                 {"reshard_tensor_start_offset", 0u}});
            in1_writer_varargs[core] = {};

            AddRuntimeArgsForNode(
                compute_run_args.runtime_arg_values, core, {{"is_worker_core", (std::uint32_t)is_worker_core}});
        } else {
            bool is_worker_core = true;
            AddRuntimeArgsForNode(
                compute_run_args.runtime_arg_values, core, {{"is_worker_core", (std::uint32_t)is_worker_core}});
        }
    }

    // Worker cores: in1 sender/writer runtime args.
    //
    // The legacy factory built one flat uint32 vector per core, with the write-back shard count
    // spliced in at a fixed index once the variable-length tail was known. That arithmetic is kept
    // verbatim below and the finished vector is split at the end: slots [0], [3]..[7] are the named
    // arguments, slots [1] and [2] held buffer addresses and are now tensor bindings, and slots
    // [8] onwards are the (bytes, noc_x, noc_y) triples the kernel walks by index.
    constexpr std::size_t num_shards_to_write_back_arg_index = 6;
    constexpr std::size_t fixed_writer_arg_count = 11;
    constexpr std::size_t writer_varargs_begin = 8;
    for (uint32_t i = 0; i < all_worker_cores_ordered.size(); ++i) {
        auto core = all_worker_cores_ordered[i];
        const auto& reader_assignment = reader_assignments[i];
        const uint32_t bank_id = reader_assignment.bank_id;

        bool is_worker_core = true;
        std::vector<uint32_t> mm_in1_sender_writer_args;
        mm_in1_sender_writer_args.push_back((std::uint32_t)is_worker_core);
        mm_in1_sender_writer_args.push_back(0u);  // [1] in1 base address -> TensorBinding
        mm_in1_sender_writer_args.push_back(0u);  // [2] bias base address -> TensorBinding

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
        mm_in1_sender_writer_args.push_back(reader_assignment.worker_index);

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

            mm_in1_sender_writer_args.insert(
                mm_in1_sender_writer_args.begin() + num_shards_to_write_back_arg_index, num_cores_write_back);
        }

        // A worker can legitimately have no output storage shard to reshard, for example when the reader count exceeds
        // the output shard count. The kernel still materializes its fixed writer-argument views before the zero-count
        // write-back loop, so provide neutral placeholders for those slots.
        if (mm_in1_sender_writer_args.size() < fixed_writer_arg_count) {
            mm_in1_sender_writer_args.resize(fixed_writer_arg_count, 0);
        }

        AddRuntimeArgsForNode(
            in1_run_args.runtime_arg_values,
            core,
            {{"is_worker_core", mm_in1_sender_writer_args[0]},
             {"dram_bank_id", mm_in1_sender_writer_args[3]},
             {"vc", mm_in1_sender_writer_args[4]},
             {"dram_reader_index", mm_in1_sender_writer_args[5]},
             {"num_shard_to_write_back", mm_in1_sender_writer_args[6]},
             {"reshard_tensor_start_offset", mm_in1_sender_writer_args[7]}});

        AdvancedKernelRunArgs::Varargs write_back_varargs(
            mm_in1_sender_writer_args.begin() + writer_varargs_begin, mm_in1_sender_writer_args.end());
        num_in1_writer_varargs = std::max<uint32_t>(num_in1_writer_varargs, write_back_varargs.size());
        in1_writer_varargs[core] = std::move(write_back_varargs);
    }

    TT_FATAL(
        total_tensor_width_written_back <= expected_max_total_width,
        "more datums written back to sharded tensor, L1 corruption, expected: {}, actual: {}",
        expected_max_total_width,
        total_tensor_width_written_back);

    // One declared vararg count for the kernel, so nodes with fewer write-back shards are padded
    // out. The kernel's loop is bounded by num_shard_to_write_back, so it never reads the padding.
    for (auto& [core, varargs] : in1_writer_varargs) {
        varargs.resize(num_in1_writer_varargs, 0u);
        in1_run_args.advanced_options.runtime_varargs[core] = std::move(varargs);
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Build KernelSpecs
    ////////////////////////////////////////////////////////////////////////////

    // in0 sender kernel (reader - RISCV_1)
    KernelSpec in0_sender{
        .unique_id = IN0_SENDER,
        .source =
            "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/"
            "reader_bmm_tile_layout_in0_sender_dram_sharded.cpp",
        .compiler_options = {.defines = KernelSpec::CompilerOptions::Defines(mm_kernel_in0_sender_define)},
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = IN0_DFB,
                    .accessor_name = "in0",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                // Sync-free one-toucher: the kernel only takes the shard's base address from it,
                // with no FIFO traffic, so it stands as both endpoints of its own buffer.
                DFBBinding{
                    .dfb_spec_name = IN0_SHARDED_DFB,
                    .accessor_name = "in0_sharded",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = IN0_SHARDED_DFB,
                    .accessor_name = "in0_sharded",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
            },
        .semaphore_bindings =
            {
                SemaphoreBinding{
                    .semaphore_spec_name = IN0_MCAST_SENDER_SEM,
                    .accessor_name = "in0_mcast_sender",
                },
                SemaphoreBinding{
                    .semaphore_spec_name = IN0_MCAST_RECEIVER_SEM,
                    .accessor_name = "in0_mcast_receiver",
                },
            },
        .compile_time_args =
            {
                {"in0_block_num_tiles", in0_block_num_tiles},
                {"in0_block_size_bytes", in0_block_num_tiles * in0_single_tile_size},
                {"in0_last_ktile_w", in0_last_ktile_w},
                {"in0_last_ktile_h", 0u},  // transpose not supported
                {"in0_mcast_num_dests", num_worker_cores},
                {"in0_mcast_num_cores", num_mcast_cores},
                {"num_blocks", num_blocks},
                {"in0_mcast_dest_noc_start_x", (std::uint32_t)start_core_noc.x},
                {"in0_mcast_dest_noc_start_y", (std::uint32_t)start_core_noc.y},
                {"in0_mcast_dest_noc_end_x", (std::uint32_t)end_core_noc.x},
                {"in0_mcast_dest_noc_end_y", (std::uint32_t)end_core_noc.y},
                {"num_blocks_per_shard", num_blocks_per_shard},
                {"in0_block_w", in0_block_w},
            },
        .runtime_arg_schema =
            {
                .runtime_arg_names = {"worker_core_type", "sender_id", "is_last_ktile_padded"},
            },
        .hw_config = DataMovementGen1Config{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = in0_noc},
        .advanced_options = {.num_runtime_varargs = num_in0_sender_varargs},
    };

    // in1 sender/writer kernel (writer - RISCV_0)
    KernelSpec in1_sender_writer{
        .unique_id = IN1_SENDER_WRITER,
        .source =
            "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/"
            "reader_bmm_tile_layout_in1_sender_dram_sharded.cpp",
        .compiler_options = {.defines = KernelSpec::CompilerOptions::Defines(mm_kernel_in1_sender_writer_defines)},
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = IN1_DFB,
                    .accessor_name = "in1",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = OUT_DFB,
                    .accessor_name = "out",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                // Sync-free one-toucher, as in0_sharded is on the sender: the kernel writes the
                // resharded output through the tensor's own L1 address with no FIFO traffic.
                DFBBinding{
                    .dfb_spec_name = OUT_RESHARD_DFB,
                    .accessor_name = "out_reshard",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = OUT_RESHARD_DFB,
                    .accessor_name = "out_reshard",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{
                    .tensor_parameter_name = IN1,
                    .accessor_name = "in1",
                },
            },
        .compile_time_args =
            {
                {"in1_page_size", in1_buffer_page_size},
                {"in1_num_pages", in1_buffer_num_pages},
                {"in1_block_w", per_core_N_compute},  // padded, used only for the bias buffer
                {"in1_block_num_tiles", per_core_N_in1_sender * in0_block_w},
                {"num_blocks", num_blocks},
                {"out_block_num_tiles", out_block_tiles},
                {"out_tensor_stride_w_bytes", per_core_N_compute * output_single_tile_size},
                {"out_reshard_tensor_stride_w_bytes", per_core_N_storage * output_single_tile_size},
                {"per_core_M", per_core_M},
                {"workers_per_bank", workers_per_bank},
                {"bank_row_stride_tiles", in1_shard_width_tiles},
                {"reader_width_tiles", per_core_N_in1_sender},
            },
        .runtime_arg_schema =
            {
                .runtime_arg_names =
                    {"is_worker_core",
                     "dram_bank_id",
                     "vc",
                     "dram_reader_index",
                     "num_shard_to_write_back",
                     "reshard_tensor_start_offset"},
            },
        .hw_config = DataMovementGen1Config{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = in1_noc},
        .advanced_options = {.num_runtime_varargs = num_in1_writer_varargs},
    };
    if (bias.has_value()) {
        in1_sender_writer.dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = BIAS_DFB,
            .accessor_name = "bias",
            .endpoint_type = DFBEndpointType::PRODUCER,
        });
        in1_sender_writer.tensor_bindings.push_back(TensorBinding{
            .tensor_parameter_name = BIAS,
            .accessor_name = "bias",
        });
        in1_sender_writer.compile_time_args.insert({"in3_page_size", bias_buffer_page_size});
        in1_sender_writer.compile_time_args.insert({"in3_num_pages", bias_buffer_num_pages});
    }

    // Compute kernel
    uint32_t in0_subblock_num_tiles = out_subblock_h * in0_block_w;
    uint32_t in1_per_core_w = per_core_N_in1_sender;
    uint32_t out_subblock_num_tiles = out_subblock_h * out_subblock_w;

    // The legacy ComputeConfigDescriptor set no unpack_to_dest_mode vector, so every buffer took
    // UnpackToDestMode::Default -- the SrcA/B path, which is UnpackMode::UnpackToSrc. Stated
    // explicitly rather than left implicit because a compute kernel that consumes a Float32
    // dataflow buffer with a 32-bit Dest register must make the choice explicit, and the
    // intermediate buffer's format is Float32 whenever fp32_dest_acc_en is set. The values are the
    // legacy ones, so the lowered per-buffer vector is unchanged.
    unpack_modes(compute_hw) = {
        {IN0_DFB, UnpackMode::UnpackToSrc},
        {IN1_DFB, UnpackMode::UnpackToSrc},
        {INTERMED0_DFB, UnpackMode::UnpackToSrc},
    };
    if (bias.has_value()) {
        unpack_modes(compute_hw).insert({BIAS_DFB, UnpackMode::UnpackToSrc});
    }

    KernelSpec compute{
        .unique_id = COMPUTE,
        .source =
            "ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/"
            "bmm_large_block_zm_fused_bias_activation_metal2.cpp",
        // Legacy ComputeConfigDescriptor defaults opt_level to O3; Metal 2.0's type-agnostic
        // CompilerOptions defaults to O2, so a compute kernel must say O3 explicitly or it
        // silently drops a level.
        .compiler_options =
            {
                .defines = KernelSpec::CompilerOptions::Defines(mm_kernel_defines),
                .opt_level = KernelBuildOptLevel::O3,
            },
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = IN0_DFB,
                    .accessor_name = "in0",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                DFBBinding{
                    .dfb_spec_name = IN1_DFB,
                    .accessor_name = "in1",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                DFBBinding{
                    .dfb_spec_name = OUT_DFB,
                    .accessor_name = "out",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                // The partials buffer is compute's alone: it packs into it and reads it back, so
                // it holds both endpoints.
                DFBBinding{
                    .dfb_spec_name = INTERMED0_DFB,
                    .accessor_name = "intermed0",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = INTERMED0_DFB,
                    .accessor_name = "intermed0",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
            },
        .compile_time_args =
            {
                {"in0_block_w", in0_block_w},
                {"in0_num_subblocks", in0_num_subblocks},
                {"in0_block_num_tiles", in0_block_num_tiles},
                {"in0_subblock_num_tiles", in0_subblock_num_tiles},
                {"in1_num_subblocks", in1_num_subblocks},
                {"in1_block_num_tiles", in1_block_tiles},
                {"in1_block_w", in1_per_core_w},
                {"num_blocks_inner_dim", num_blocks},
                {"num_blocks_w_dim", 1u},
                {"num_blocks_h_dim", 1u},
                {"out_subblock_h", out_subblock_h},
                {"out_subblock_w", out_subblock_w},
                {"out_subblock_num_tiles", out_subblock_num_tiles},
                {"batch", B},
                {"out_block_num_tiles", out_block_tiles},
                {"untilize_out", (std::uint32_t)untilize_out},
                {"get_batch_from_reader", 0u},
                {"bias_ntiles", per_core_N_compute},
                {"last_subblock_w_valid", last_subblock_w_valid},
            },
        .runtime_arg_schema =
            {
                .runtime_arg_names = {"is_worker_core"},
            },
        .hw_config = std::move(compute_hw),
    };
    if (bias.has_value()) {
        compute.dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = BIAS_DFB,
            .accessor_name = "bias",
            .endpoint_type = DFBEndpointType::CONSUMER,
        });
        compute.compile_time_args.insert({"row_broadcast_bias", row_broadcast_bias ? 1u : 0u});
    }
    if (fused_activation.has_value() && fused_activation.value().op_type != UnaryOpType::RELU) {
        using ttnn::operations::matmul::utilities::get_activation_params;
        const auto params = get_activation_params(fused_activation.value());
        compute.compile_time_args.insert({"activation_type", static_cast<uint32_t>(params.type)});
        compute.compile_time_args.insert({"activation_param0", params.param0});
        compute.compile_time_args.insert({"activation_param1", params.param1});
        compute.compile_time_args.insert({"activation_param2", params.param2});
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Assemble
    ////////////////////////////////////////////////////////////////////////////

    Group<KernelSpec> kernels;
    kernels.reserve(3);
    kernels.push_back(std::move(in0_sender));
    kernels.push_back(std::move(in1_sender_writer));
    kernels.push_back(std::move(compute));

    Group<TensorParameter> tensor_parameters;
    tensor_parameters.reserve(bias.has_value() ? 4 : 3);
    tensor_parameters.push_back(TensorParameter{.unique_id = IN0, .spec = in0_tensor.tensor_spec()});
    tensor_parameters.push_back(TensorParameter{.unique_id = IN1, .spec = in1_tensor.tensor_spec()});
    tensor_parameters.push_back(TensorParameter{.unique_id = OUTPUT, .spec = out_tensor.tensor_spec()});
    if (bias.has_value()) {
        tensor_parameters.push_back(TensorParameter{.unique_id = BIAS, .spec = bias->tensor_spec()});
    }

    ProgramSpec spec{
        .name = "matmul_multi_core_reuse_mcast_dram_sharded",
        .kernels = std::move(kernels),
        .dataflow_buffers = std::move(dataflow_buffers),
        .semaphores = std::move(semaphores),
        .tensor_parameters = std::move(tensor_parameters),
        .work_units =
            {
                WorkUnitSpec{
                    .name = "all_cores_in_rect_grid",
                    .kernels = {IN0_SENDER, IN1_SENDER_WRITER, COMPUTE},
                    .target_nodes = all_cores_in_rect_grid,
                },
            },
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args.reserve(3);
    run_args.kernel_run_args.push_back(std::move(in0_run_args));
    run_args.kernel_run_args.push_back(std::move(in1_run_args));
    run_args.kernel_run_args.push_back(std::move(compute_run_args));
    run_args.tensor_args = {
        {IN0, in0_tensor},
        {IN1, in1_tensor},
        {OUTPUT, out_tensor},
    };
    if (bias.has_value()) {
        run_args.tensor_args.insert({BIAS, *bias});
    }

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

}  // namespace reuse_dram_sharded_optimized_helpers

ttnn::device_operation::ProgramArtifacts
MatmulMultiCoreReuseMultiCastDRAMShardedProgramFactory::create_program_artifacts(
    const ttnn::prim::MatmulParams& operation_attributes,
    const ttnn::prim::MatmulInputs& tensor_args,
    std::vector<ttnn::Tensor>& tensor_return_value) {
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

    // Dataflow buffer dataformats
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

    const bool row_broadcast_bias = operations::matmul::utilities::fused_matmul_bias_row_broadcastable(bias);

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
    const auto& program_config = std::get<operations::matmul::MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig>(
        operation_attributes.program_config.value());
    const auto& in0_block_w = program_config.in0_block_w;
    const auto& per_core_M = program_config.per_core_M;
    const auto& per_core_N = program_config.per_core_N;
    const auto& workers_per_bank = program_config.num_workers_per_dram_bank;
    const auto& fused_activation = program_config.fused_activation;

    const auto& untilize_out = operation_attributes.untilize_out;
    const bool skip_compute = false;
    const bool skip_in0_mcast = false;
    const bool skip_write_back = false;

    // math_fidelity, math_approx_mode and dst_full_sync_en fed the legacy compute config and
    // nothing else; the resolved ComputeHardwareConfig built at the call below now carries them,
    // so only fp32_dest_acc_en and packer_l1_acc are still read here.
    [[maybe_unused]] auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);

    uint32_t B = 1;
    uint32_t Mt = get_batch_size(ashape) * ashape[-2] / in0_tile_shape[0];
    uint32_t Kt = ashape[-1] / in0_tile_shape[1];
    uint32_t Nt = bshape[-1] / in1_tile_shape[1];
    uint32_t in0_last_ktile_w = a.logical_shape()[-1] % in0_tile_shape[1];

    TT_FATAL(Kt % in0_block_w == 0, "Kt ({}) must be divisible by in0_block_w ({})", Kt, in0_block_w);

    const auto& output_mesh = output.mesh_tensor();

    return reuse_dram_sharded_optimized_helpers::create_program_dram_sharded_spec(
        device,
        input_all_cores_storage,
        output_all_cores_storage,
        // math_fidelity / math_approx_mode / dst_full_sync_en fed the legacy compute config and
        // nothing else, so the resolved hardware config carries them from here on.
        ttnn::to_compute_hardware_config(device->arch(), compute_kernel_config),
        fp32_dest_acc_en,
        packer_l1_acc,
        ttnn::get_throttle_level(operation_attributes.compute_kernel_config),
        B,
        Mt,
        Nt,
        Kt,
        in0_block_w,
        in0_last_ktile_w,
        per_core_M,
        per_core_N,
        workers_per_bank,
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

}  // namespace ttnn::prim

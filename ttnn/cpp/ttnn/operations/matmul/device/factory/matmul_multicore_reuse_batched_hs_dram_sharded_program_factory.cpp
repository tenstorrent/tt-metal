// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/matmul/device/factory/matmul_multicore_reuse_batched_hs_dram_sharded_program_factory.hpp"
#include "ttnn/operations/matmul/device/utilities/matmul_utilities.hpp"
#include "ttnn/operations/matmul/device/config/matmul_program_config.hpp"

#include <algorithm>
#include <map>
#include <set>
#include <utility>

#include "hostdevcommon/common_values.hpp"
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "ttnn/operations/compute_throttle_utils.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/operations/matmul/shared_with_host/activation_type.hpp"

using namespace tt;

using ttnn::operations::unary::UnaryOpType;
using ttnn::operations::unary::UnaryWithParam;

using tt::tt_metal::CBDescriptor;
using tt::tt_metal::CBFormatDescriptor;
using tt::tt_metal::ComputeConfigDescriptor;
using tt::tt_metal::DataMovementConfigDescriptor;
using tt::tt_metal::KernelDescriptor;
using tt::tt_metal::ProgramDescriptor;

namespace ttnn::prim {
namespace reuse_batched_hs_dram_sharded_optimized_helpers {

using dram_sharded_helpers::get_max_page_size_and_num_pages;
using dram_sharded_helpers::get_optimal_dram_bank_to_reader_assignment;

// Batch-sharded DRAM matmul
// For batched matmul: [1, B, M, K] x [1, B, K, N] = [1, B, M, N]
// Sharded by batch dimension - each worker handles B/num_workers complete matmuls
// ProgramDescriptor variant: translates the same logic as create_program_batch_sharded
// into a lightweight ProgramDescriptor (no Program object created).
static ProgramDescriptor create_program_batch_sharded_descriptor(
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
    uint32_t K,
    uint32_t /* N */,
    uint32_t in0_block_w,
    uint32_t per_core_M,
    uint32_t per_core_N,
    std::optional<UnaryWithParam> fused_activation,
    const tt_metal::MeshTensor& in0_tensor,
    const tt_metal::MeshTensor& in1_tensor,
    ttsl::optional_reference<const tt_metal::MeshTensor> bias_tensor,
    const tt_metal::MeshTensor& out_tensor,
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
    bool skip_write_back) {
    tt_metal::NOC in1_noc = tt::tt_metal::detail::preferred_noc_for_dram_read(device->arch());

    std::vector<CoreCoord> all_worker_cores_ordered;
    CoreRangeSet all_worker_cores;
    get_optimal_dram_bank_to_reader_assignment(device, all_worker_cores_ordered, all_worker_cores, in1_noc);

    // Input / output storage core ordering
    std::vector<CoreCoord> input_storage_cores_ordered =
        corerange_to_cores(input_all_storage_cores, std::nullopt, true);
    std::vector<CoreCoord> output_storage_cores_ordered =
        corerange_to_cores(output_all_storage_cores, std::nullopt, true);

    uint32_t num_workers = all_worker_cores_ordered.size();
    TT_FATAL(
        input_storage_cores_ordered.size() == num_workers,
        "Input storage cores ({}) must match number of workers/DRAM banks ({})",
        input_storage_cores_ordered.size(),
        num_workers);
    TT_FATAL(
        output_storage_cores_ordered.size() == num_workers,
        "Output storage cores ({}) must match number of workers/DRAM banks ({})",
        output_storage_cores_ordered.size(),
        num_workers);
    for (uint32_t i = 0; i < num_workers; ++i) {
        TT_FATAL(
            input_storage_cores_ordered[i] == all_worker_cores_ordered[i],
            "Input storage core ordering mismatch at index {}",
            i);
        TT_FATAL(
            output_storage_cores_ordered[i] == all_worker_cores_ordered[i],
            "Output storage core ordering mismatch at index {}",
            i);
    }

    // NOC coordinate vectors for storage cores
    std::vector<uint32_t> input_storage_noc_x, input_storage_noc_y;
    std::vector<uint32_t> output_storage_noc_x, output_storage_noc_y;
    for (const auto& core : input_storage_cores_ordered) {
        auto phys_core = device->worker_core_from_logical_core(core);
        input_storage_noc_x.push_back(phys_core.x);
        input_storage_noc_y.push_back(phys_core.y);
    }
    for (const auto& core : output_storage_cores_ordered) {
        auto phys_core = device->worker_core_from_logical_core(core);
        output_storage_noc_x.push_back(phys_core.x);
        output_storage_noc_y.push_back(phys_core.y);
    }

    // Bounding box of all cores (workers + storage)
    std::set<CoreRange> all_cores_set;
    for (const auto& core : all_worker_cores_ordered) {
        all_cores_set.insert(CoreRange(core));
    }
    for (const auto& core : input_storage_cores_ordered) {
        all_cores_set.insert(CoreRange(core));
    }
    for (const auto& core : output_storage_cores_ordered) {
        all_cores_set.insert(CoreRange(core));
    }
    CoreRangeSet all_cores(all_cores_set);
    CoreRange bounding_box = all_cores.bounding_box();
    CoreRangeSet all_cores_in_rect_grid({bounding_box});

    uint32_t num_cores = num_workers;
    uint32_t num_dram_banks = device->num_dram_channels();
    uint32_t batches_per_core = (B + num_cores - 1) / num_cores;

    TT_FATAL(
        num_cores <= num_dram_banks,
        "Number of worker cores ({}) cannot exceed number of DRAM banks ({})",
        num_cores,
        num_dram_banks);

    // Subblock parameters
    auto subblock_hw = operations::matmul::bmm_op_utils::get_matmul_subblock_params(
        per_core_M, per_core_N, false, false, fp32_dest_acc_en);
    auto out_subblock_h = std::get<0>(subblock_hw);
    auto out_subblock_w = std::get<1>(subblock_hw);

    uint32_t num_blocks = K / in0_block_w;
    bool packer_l1_acc_en = packer_l1_acc && num_blocks > 1;

    tt::DataFormat interm0_data_format = packer_l1_acc_en
                                             ? (fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b)
                                             : (fp32_dest_acc_en ? tt::DataFormat::Float32 : output_data_format);

    // Tile sizes
    uint32_t in0_single_tile_size = in0_tile.get_tile_size(in0_data_format);
    uint32_t in1_single_tile_size = in1_tile.get_tile_size(in1_data_format);
    uint32_t bias_single_tile_size = bias_tile.get_tile_size(bias_data_format);
    uint32_t output_single_tile_size = output_tile.get_tile_size(output_data_format);
    uint32_t interm0_single_tile_size = output_tile.get_tile_size(interm0_data_format);

    // CB sizes
    uint32_t in0_block_tiles = per_core_M * in0_block_w;
    uint32_t in0_CB_tiles = in0_block_tiles * 2;
    uint32_t in0_CB_size = in0_CB_tiles * in0_single_tile_size;

    uint32_t in1_block_tiles = in0_block_w * per_core_N;
    uint32_t in1_CB_tiles = in1_block_tiles * 3;
    uint32_t in1_CB_size = in1_CB_tiles * in1_single_tile_size;

    uint32_t out_block_tiles = per_core_M * per_core_N;
    uint32_t interm0_CB_size = out_block_tiles * interm0_single_tile_size;

    uint32_t in0_shard_tiles = in0_tensor.shard_spec()->shape[0] / in0_tile.get_tile_shape()[0] *
                               in0_tensor.shard_spec()->shape[1] / in0_tile.get_tile_shape()[1];
    uint32_t in2_CB_size = in0_shard_tiles * in0_single_tile_size;

    uint32_t in3_block_tiles = per_core_N;
    uint32_t in3_CB_size = in3_block_tiles * bias_single_tile_size;

    uint32_t out_shard_tiles = out_tensor.shard_spec()->shape[0] / output_tile.get_tile_shape()[0] *
                               out_tensor.shard_spec()->shape[1] / output_tile.get_tile_shape()[1];
    uint32_t out_reshard_CB_size = out_shard_tiles * output_single_tile_size;

    // Page sizes for DRAM reads
    uint32_t in1_buffer_page_size, in1_buffer_num_pages;
    get_max_page_size_and_num_pages(
        device, in1_block_tiles, in1_single_tile_size, in1_buffer_page_size, in1_buffer_num_pages);

    uint32_t bias_buffer_page_size, bias_buffer_num_pages;
    get_max_page_size_and_num_pages(
        device, in3_block_tiles, bias_single_tile_size, bias_buffer_page_size, bias_buffer_num_pages);

    // Tensor stride calculations
    uint32_t in0_batch_stride_bytes = per_core_M * K * in0_single_tile_size;
    uint32_t in1_batch_stride_bytes = K * per_core_N * in1_single_tile_size;
    uint32_t out_batch_stride_bytes = per_core_M * per_core_N * output_single_tile_size;

    ////////////////////////////////////////////////////////////////////////////
    //                      Build CBDescriptors
    ////////////////////////////////////////////////////////////////////////////
    ProgramDescriptor desc;

    tt::tt_metal::TileDescriptor in0_tile_desc{in0_tile};
    tt::tt_metal::TileDescriptor in1_tile_desc{in1_tile};
    tt::tt_metal::TileDescriptor bias_tile_desc{bias_tile};
    tt::tt_metal::TileDescriptor output_tile_desc{output_tile};

    // CB 0: in0 (activations) - on all cores in bounding box
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

    // CB 1: in1 (weights) - on all cores in bounding box
    {
        CBDescriptor cb_desc;
        cb_desc.total_size = in1_CB_size;
        cb_desc.core_ranges = all_cores_in_rect_grid;
        cb_desc.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_1,
            .data_format = in1_data_format,
            .page_size = in1_single_tile_size,
            .tile = in1_tile_desc});
        desc.cbs.push_back(std::move(cb_desc));
    }

    // CB 2: sharded in0 buffer - on INPUT storage cores, backed by in0_buffer
    {
        CBDescriptor cb_desc;
        cb_desc.total_size = in2_CB_size;
        cb_desc.core_ranges = input_all_storage_cores;
        cb_desc.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_2,
            .data_format = in0_data_format,
            .page_size = in0_single_tile_size,
            .tile = in0_tile_desc});
        cb_desc.tensor = &in0_tensor;
        desc.cbs.push_back(std::move(cb_desc));
    }

    // CB 3: bias (if fused) - on all cores in bounding box
    if (bias_tensor.has_value()) {
        CBDescriptor cb_desc;
        cb_desc.total_size = in3_CB_size;
        cb_desc.core_ranges = all_cores_in_rect_grid;
        cb_desc.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_3,
            .data_format = bias_data_format,
            .page_size = bias_single_tile_size,
            .tile = bias_tile_desc});
        desc.cbs.push_back(std::move(cb_desc));
    }

    // CB 4 & 5: output and intermediate - on worker cores
    uint32_t output_cb_index = tt::CBIndex::c_4;
    uint32_t interm0_cb_index = tt::CBIndex::c_5;
    if (interm0_data_format != output_data_format) {
        // Separate CBs for output and intermediate
        {
            CBDescriptor cb_desc;
            cb_desc.total_size = out_reshard_CB_size;
            cb_desc.core_ranges = all_worker_cores;
            cb_desc.format_descriptors.push_back(CBFormatDescriptor{
                .buffer_index = output_cb_index,
                .data_format = output_data_format,
                .page_size = output_single_tile_size,
                .tile = output_tile_desc});
            desc.cbs.push_back(std::move(cb_desc));
        }
        {
            CBDescriptor cb_desc;
            cb_desc.total_size = interm0_CB_size;
            cb_desc.core_ranges = all_worker_cores;
            cb_desc.format_descriptors.push_back(CBFormatDescriptor{
                .buffer_index = interm0_cb_index,
                .data_format = interm0_data_format,
                .page_size = interm0_single_tile_size,
                .tile = output_tile_desc});
            desc.cbs.push_back(std::move(cb_desc));
        }
    } else {
        // Output and intermediate share the same buffer
        CBDescriptor cb_desc;
        cb_desc.total_size = out_reshard_CB_size;
        cb_desc.core_ranges = all_worker_cores;
        cb_desc.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = output_cb_index,
            .data_format = output_data_format,
            .page_size = output_single_tile_size,
            .tile = output_tile_desc});
        cb_desc.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = interm0_cb_index,
            .data_format = interm0_data_format,
            .page_size = interm0_single_tile_size,
            .tile = output_tile_desc});
        desc.cbs.push_back(std::move(cb_desc));
    }

    // CB 6: output reshard buffer - on OUTPUT storage cores, backed by out_buffer
    {
        CBDescriptor cb_desc;
        cb_desc.total_size = out_reshard_CB_size;
        cb_desc.core_ranges = output_all_storage_cores;
        cb_desc.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_6,
            .data_format = output_data_format,
            .page_size = output_single_tile_size,
            .tile = output_tile_desc});
        cb_desc.tensor = &out_tensor;
        desc.cbs.push_back(std::move(cb_desc));
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Kernel defines
    ////////////////////////////////////////////////////////////////////////////
    std::map<std::string, std::string> mm_kernel_defines;
    std::map<std::string, std::string> reader_defines;
    std::map<std::string, std::string> writer_defines;

    if (bias_tensor.has_value()) {
        mm_kernel_defines["FUSE_BIAS"] = "1";
        writer_defines["FUSE_BIAS"] = "1";
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
    if (skip_compute) {
        mm_kernel_defines["SKIP_COMPUTE"] = "1";
    }
    if (skip_write_back) {
        writer_defines["SKIP_WRITE_BACK"] = "1";
    }
    mm_kernel_defines["MATMUL_DRAM_SHARDED"] = "1";

    ttnn::operations::compute_throttle_utils::add_stagger_defines_if_needed(
        device->arch(), num_cores, mm_kernel_defines);
    ttnn::operations::compute_throttle_utils::throttle_mm_perf(
        device->arch(), num_cores, mm_kernel_defines, throttle_level);

    // Helper to convert std::map defines to KernelDescriptor::Defines
    auto map_to_defines = [](const std::map<std::string, std::string>& m) -> KernelDescriptor::Defines {
        KernelDescriptor::Defines result;
        result.reserve(m.size());
        for (const auto& [k, v] : m) {
            result.emplace_back(k, v);
        }
        return result;
    };

    ////////////////////////////////////////////////////////////////////////////
    //                      Compile-time args
    ////////////////////////////////////////////////////////////////////////////
    std::vector<uint32_t> in0_reader_compile_args = {
        (uint32_t)in0_block_tiles,
        (uint32_t)in0_block_tiles * in0_single_tile_size,
        (uint32_t)num_blocks,
        (uint32_t)batches_per_core,
        (uint32_t)in0_batch_stride_bytes,
        (uint32_t)in2_CB_size,
    };

    std::vector<uint32_t> in1_writer_compile_args = {
        (uint32_t)in1_buffer_page_size,
        (uint32_t)in1_buffer_num_pages,
        (uint32_t)per_core_N,
        (uint32_t)in1_block_tiles,
        (uint32_t)num_blocks,
        (uint32_t)out_block_tiles,
        (uint32_t)batches_per_core,
        (uint32_t)in1_batch_stride_bytes,
        (uint32_t)out_batch_stride_bytes,
        (uint32_t)out_reshard_CB_size,
    };
    if (bias_tensor.has_value()) {
        in1_writer_compile_args.push_back(bias_buffer_page_size);
        in1_writer_compile_args.push_back(bias_buffer_num_pages);
        in1_writer_compile_args.push_back(in3_block_tiles);
    }

    uint32_t in0_num_subblocks = per_core_M / out_subblock_h;
    uint32_t in1_num_subblocks = per_core_N / out_subblock_w;
    uint32_t in0_subblock_num_tiles = out_subblock_h * in0_block_w;
    uint32_t out_subblock_num_tiles = out_subblock_h * out_subblock_w;

    std::vector<uint32_t> compute_kernel_args = {
        in0_block_w,
        in0_num_subblocks,
        in0_block_tiles,
        in0_subblock_num_tiles,
        in1_num_subblocks,
        in1_block_tiles,
        per_core_N,
        num_blocks,
        1,
        1,
        out_subblock_h,
        out_subblock_w,
        out_subblock_num_tiles,
        batches_per_core,
        out_block_tiles,
        untilize_out ? 1u : 0u,
        0u,
        0u,
    };
    if (bias_tensor.has_value()) {
        compute_kernel_args.push_back(1u);  // row_broadcast_bias: DRAM sharded always uses row broadcast
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Build Kernel Descriptors
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::NOC in0_noc = tt::tt_metal::detail::preferred_noc_for_dram_write(device->arch());

    writer_defines["OUT_SHARDED"] = "1";

    // in0 reader kernel
    KernelDescriptor in0_reader_kernel_desc;
    in0_reader_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/"
        "reader_bmm_tile_layout_in0_sender_dram_sharded_height.cpp";
    in0_reader_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    in0_reader_kernel_desc.core_ranges = all_cores_in_rect_grid;
    in0_reader_kernel_desc.compile_time_args = in0_reader_compile_args;
    in0_reader_kernel_desc.defines = map_to_defines(reader_defines);
    in0_reader_kernel_desc.named_compile_time_args = {
        {"cb_in0", tt::CBIndex::c_0},
    };
    in0_reader_kernel_desc.config =
        DataMovementConfigDescriptor{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = in0_noc};

    // in1 writer kernel
    KernelDescriptor in1_writer_kernel_desc;
    in1_writer_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/"
        "reader_bmm_tile_layout_in1_sender_dram_sharded_height.cpp";
    in1_writer_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    in1_writer_kernel_desc.core_ranges = all_cores_in_rect_grid;
    in1_writer_kernel_desc.compile_time_args = in1_writer_compile_args;
    in1_writer_kernel_desc.defines = map_to_defines(writer_defines);
    in1_writer_kernel_desc.named_compile_time_args = {
        {"cb_in1", tt::CBIndex::c_1},
        {"cb_bias", tt::CBIndex::c_3},
        {"cb_out", tt::CBIndex::c_4},
    };
    in1_writer_kernel_desc.config =
        DataMovementConfigDescriptor{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = in1_noc};

    // compute kernel
    KernelDescriptor compute_kernel_desc;
    compute_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp";
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
            {"bias_ntiles", per_core_N},
            // This factory does not pad per_core_N_compute beyond per_core_N_in1_sender, so the
            // last subblock is always fully valid. Pass out_subblock_w so the compute kernel takes
            // its original full-width path (last_subblock_padded == false).
            {"last_subblock_w_valid", out_subblock_w},
        };
        if (fused_activation.has_value() && fused_activation.value().op_type != UnaryOpType::RELU) {
            using ttnn::operations::matmul::utilities::get_activation_params;
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
        .math_approx_mode = math_approx_mode};

    ////////////////////////////////////////////////////////////////////////////
    //                      Runtime Args (per-core loop)
    ////////////////////////////////////////////////////////////////////////////
    std::vector<CoreCoord> all_cores_in_rect_grid_vec = corerange_to_cores(all_cores_in_rect_grid);
    std::set<CoreCoord> worker_cores_set(all_worker_cores_ordered.begin(), all_worker_cores_ordered.end());

    std::vector<uint32_t> bank_ids;

    // Idle cores in the bounding box
    for (const auto& core : all_cores_in_rect_grid_vec) {
        bool is_worker = worker_cores_set.contains(core);

        if (!is_worker) {
            std::vector<uint32_t> in0_idle_args = {0u};
            in0_reader_kernel_desc.runtime_args.emplace_back(core, in0_idle_args);

            std::vector<uint32_t> in1_idle_args = {0u};
            in1_writer_kernel_desc.runtime_args.emplace_back(core, in1_idle_args);

            std::vector<uint32_t> compute_idle_args = {0u};
            compute_kernel_desc.runtime_args.emplace_back(core, compute_idle_args);
        }
    }

    // Worker cores
    for (uint32_t worker_idx = 0; worker_idx < all_worker_cores_ordered.size(); ++worker_idx) {
        auto core = all_worker_cores_ordered[worker_idx];

        uint32_t bank_id = worker_idx;
        uint32_t vc = bank_id & 0x3;
        bank_ids.push_back(bank_id);
        for (uint32_t j = 0; j < worker_idx; ++j) {
            auto core_prev = all_worker_cores_ordered[j];
            if (core_prev.y == core.y && ((bank_id & 0x3) == (bank_ids[j] & 0x3))) {
                vc = (vc + 1) & 0x3;
                break;
            }
        }

        // in0 reader runtime args
        KernelDescriptor::RTArgList in0_reader_args;
        in0_reader_args.push_back(uint32_t{1u});
        in0_reader_args.push_back(uint32_t{input_storage_noc_x[worker_idx]});
        in0_reader_args.push_back(uint32_t{input_storage_noc_y[worker_idx]});
        in0_reader_args.push_back(in0_tensor);
        in0_reader_kernel_desc.emplace_runtime_args(core, in0_reader_args);

        // in1 writer runtime args
        KernelDescriptor::RTArgList in1_writer_args;
        in1_writer_args.push_back(uint32_t{1u});
        in1_writer_args.push_back(in1_tensor);
        if (bias_tensor.has_value()) {
            in1_writer_args.push_back(*bias_tensor);
        } else {
            in1_writer_args.push_back(uint32_t{0u});
        }
        in1_writer_args.push_back(uint32_t{bank_id});
        in1_writer_args.push_back(uint32_t{vc});
        in1_writer_args.push_back(uint32_t{output_storage_noc_x[worker_idx]});
        in1_writer_args.push_back(uint32_t{output_storage_noc_y[worker_idx]});
        in1_writer_args.push_back(out_tensor);
        in1_writer_kernel_desc.emplace_runtime_args(core, in1_writer_args);

        // Compute runtime args
        std::vector<uint32_t> compute_runtime_args = {
            1u,
        };
        compute_kernel_desc.runtime_args.emplace_back(core, compute_runtime_args);
    }

    // Push all kernel descriptors
    desc.kernels.push_back(std::move(in0_reader_kernel_desc));
    desc.kernels.push_back(std::move(in1_writer_kernel_desc));
    desc.kernels.push_back(std::move(compute_kernel_desc));

    return desc;
}

}  // namespace reuse_batched_hs_dram_sharded_optimized_helpers

ProgramDescriptor MatmulMultiCoreReuseBatchedHSDRAMShardedProgramFactory::create_descriptor(
    const ttnn::prim::MatmulParams& operation_attributes,
    const ttnn::prim::MatmulInputs& tensor_args,
    std::vector<ttnn::Tensor>& tensor_return_value,
    const std::optional<CoreRangeSet>& /*core_range_set*/) {
    const auto& input_tensors = tensor_args.input_tensors;
    const auto& optional_input_tensors = tensor_args.optional_input_tensors;
    const auto& output_tensors = tensor_return_value;

    const auto& a = input_tensors.at(0).mesh_tensor();
    const auto& b = input_tensors.at(1).mesh_tensor();
    auto bias = tt::tt_metal::as_optional_mesh_tensor(optional_input_tensors.at(0));
    const auto& output = output_tensors.at(0).mesh_tensor();
    const auto& ashape = a.padded_shape();
    const auto& bshape = b.padded_shape();
    auto in0_tile = a.tensor_spec().tile();
    auto in1_tile = b.tensor_spec().tile();
    auto in0_tile_shape = in0_tile.get_tile_shape();
    auto in1_tile_shape = in1_tile.get_tile_shape();
    auto output_tile = tt::tt_metal::Tile({in0_tile.get_tile_shape()[0], in1_tile.get_tile_shape()[1]});

    // CB dataformats
    tt::DataFormat in0_data_format = tt_metal::datatype_to_dataformat_converter(a.dtype());
    tt::DataFormat in1_data_format = tt_metal::datatype_to_dataformat_converter(b.dtype());
    tt::DataFormat output_data_format = tt_metal::datatype_to_dataformat_converter(output.dtype());

    tt::DataFormat bias_data_format = tt::DataFormat::Bfp8_b;
    if (bias.has_value()) {
        const auto& c = bias.value();
        TT_FATAL(&a.device() == &c.device(), "Operands to matmul need to be on the same device!");
        bias_data_format = tt_metal::datatype_to_dataformat_converter(c.dtype());
    }

    tt::tt_metal::IDevice* device = &a.mutable_device();

    TT_FATAL(
        a.shard_spec().has_value() && output.shard_spec().has_value(), "Both input A and output must have shard specs");
    CoreRangeSet input_all_cores_storage = a.shard_spec().value().grid;
    CoreRangeSet output_all_cores_storage = output.shard_spec().value().grid;

    uint32_t in0_single_tile_size = in0_tile.get_tile_size(tt_metal::datatype_to_dataformat_converter(a.dtype()));
    uint32_t in1_single_tile_size = in1_tile.get_tile_size(tt_metal::datatype_to_dataformat_converter(b.dtype()));

    TT_FATAL(
        a.mesh_buffer().device_local_size() % in0_single_tile_size == 0,
        "Input A buffer size ({}) must be divisible by single tile size ({})",
        a.mesh_buffer().device_local_size(),
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
    TT_FATAL(ashape[-2] % in0_tile_shape[0] == 0, "A.shape[-2] must be divisible by tile shape[0]");
    TT_FATAL(ashape[-1] % in0_tile_shape[1] == 0, "A.shape[-1] must be divisible by tile shape[1]");
    TT_FATAL(bshape[-2] % in1_tile_shape[0] == 0, "B.shape[-2] must be divisible by tile shape[0]");
    TT_FATAL(bshape[-1] % in1_tile_shape[1] == 0, "B.shape[-1] must be divisible by tile shape[1]");

    const auto& compute_kernel_config = operation_attributes.compute_kernel_config.value();
    const auto& program_config =
        std::get<operations::matmul::MatmulMultiCoreReuseMultiCastBatchedDRAMShardedProgramConfig>(
            operation_attributes.program_config.value());
    const auto& in0_block_w = program_config.in0_block_w;
    const auto& per_core_M = program_config.per_core_M;
    const auto& per_core_N = program_config.per_core_N;
    const auto& fused_activation = program_config.fused_activation;
    const auto& untilize_out = operation_attributes.untilize_out;

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);

    uint32_t B = ashape[1];
    uint32_t M = ashape[-2] / in0_tile_shape[0];
    uint32_t K = ashape[-1] / in0_tile_shape[1];
    uint32_t N = bshape[-1] / in1_tile_shape[1];

    TT_FATAL(per_core_M == M, "For batch sharding, per_core_M ({}) must equal M ({})", per_core_M, M);
    TT_FATAL(per_core_N == N, "For batch sharding, per_core_N ({}) must equal N ({})", per_core_N, N);
    TT_FATAL(K % in0_block_w == 0, "K ({}) must be divisible by in0_block_w ({})", K, in0_block_w);

    return reuse_batched_hs_dram_sharded_optimized_helpers::create_program_batch_sharded_descriptor(
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
        M,
        K,
        N,
        in0_block_w,
        per_core_M,
        per_core_N,
        fused_activation,
        a,
        b,
        bias,
        output,
        in0_tile,
        in1_tile,
        bias.has_value() ? bias->tensor_spec().tile() : output_tile,
        output_tile,
        in0_data_format,
        in1_data_format,
        bias_data_format,
        output_data_format,
        untilize_out,
        false,   // skip_compute
        false);  // skip_write_back
}

}  // namespace ttnn::prim

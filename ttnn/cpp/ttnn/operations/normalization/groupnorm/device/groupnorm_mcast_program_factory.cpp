// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "groupnorm_device_operation.hpp"
#include "groupnorm_program_utils.hpp"
#include "kernels/groupnorm_constants.hpp"

#include <bit>
#include <map>
#include <string>
#include <optional>
#include <unordered_map>

#include <tt-metalium/circular_buffer_constants.h>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/operations/math.hpp"

using uint32_t = std::uint32_t;
using namespace tt::tt_metal;

namespace ttnn::prim {

namespace {

KernelDescriptor::NamedCompileTimeArgs to_named_args_mcast(const std::unordered_map<std::string, uint32_t>& m) {
    KernelDescriptor::NamedCompileTimeArgs out;
    out.reserve(m.size());
    for (const auto& [k, v] : m) {
        out.emplace_back(k, v);
    }
    return out;
}

}  // namespace

tt::tt_metal::ProgramDescriptor GroupNormDeviceOperation::GroupNormMcastProgramFactory::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    const auto& a = tensor_args.input;
    const auto& gamma = tensor_args.gamma;
    const auto& beta = tensor_args.beta;
    const auto& input_mask = tensor_args.input_mask;
    const auto& reciprocals = tensor_args.reciprocals;
    auto& output = tensor_return_value;

    const uint32_t tile_height = a.tensor_spec().tile().get_height();
    const uint32_t tile_width = a.tensor_spec().tile().get_width();
    const uint32_t tile_hw = a.tensor_spec().tile().get_tile_hw();

    const auto& program_config = std::get<GroupNormMultiCoreProgramConfig>(operation_attributes.program_config);
    float eps = operation_attributes.eps;
    uint32_t num_groups = operation_attributes.num_groups;
    uint32_t num_batches = a.padded_shape()[0];
    DataType im_data_format = program_config.im_data_format;
    CoreCoord grid_size = program_config.compute_with_storage_grid_size;
    uint32_t num_out_blocks = program_config.num_out_blocks;
    bool use_welford = operation_attributes.use_welford;
    const auto& compute_kernel_config = operation_attributes.compute_kernel_config;

    if (gamma.has_value()) {
        TT_FATAL(gamma.value().layout() == Layout::ROW_MAJOR, "Gamma tensor must have ROW_MAJOR layout");
    }
    if (beta.has_value()) {
        TT_FATAL(beta.value().layout() == Layout::ROW_MAJOR, "Beta tensor must have ROW_MAJOR layout");
    }

    uint32_t groupnorm_mode = static_cast<uint32_t>(
        reciprocals.has_value() ? GroupNormMode::WELFORD_RECIPROCALS
        : use_welford           ? GroupNormMode::WELFORD_NATIVE
                                : GroupNormMode::LEGACY);
    uint32_t num_reciprocals = reciprocals.has_value() ? reciprocals.value().shard_spec().value().numel() : 0;

    tt::DataFormat in_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());
    tt::DataFormat out_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(im_data_format);
    tt::DataFormat gamma_beta_cb_data_format = tt::DataFormat::Float16_b;
    tt::DataFormat reciprocal_cb_data_format =
        reciprocals.has_value() ? tt::tt_metal::datatype_to_dataformat_converter(reciprocals.value().dtype())
                                : tt::DataFormat::Float32;
    if (gamma.has_value()) {
        gamma_beta_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(gamma.value().dtype());
    }
    if (beta.has_value()) {
        gamma_beta_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(beta.value().dtype());
    }
    tt::DataFormat in_mask_cb_data_format =
        input_mask.has_value() ? tt::tt_metal::datatype_to_dataformat_converter(input_mask.value().dtype())
                               : tt::DataFormat::Float16_b;
    uint32_t datum_size_bytes = 2;

    TT_FATAL(
        out_data_format == in_data_format,
        "input: {} and output: {} must be the same data format",
        in_data_format,
        out_data_format);

    uint32_t in_single_tile_size = tt::tile_size(in_data_format);
    uint32_t single_tile_size = tt::tile_size(cb_data_format);
    uint32_t out_single_tile_size = tt::tile_size(out_data_format);
    uint32_t gamma_beta_single_tile_size = tt::tile_size(gamma_beta_cb_data_format);
    uint32_t in_mask_single_tile_size = tt::tile_size(in_mask_cb_data_format);

    IDevice* device = a.device();

    const auto& shape = a.padded_shape();
    uint32_t H = shape[1] * shape[2] * num_batches;
    uint32_t Ht = H / tile_height;
    uint32_t W = shape[3];
    uint32_t Wt = W / tile_width;

    TT_FATAL(W % tile_width == 0, "W (channels): {} must be divisible by {}", W, tile_width);
    TT_FATAL(W % num_groups == 0, "W (channels): {} must be divisible by num_groups: {}", W, num_groups);
    uint32_t num_virtual_cols = std::min<uint32_t>(grid_size.x, num_groups);
    while ((W / num_virtual_cols) % tile_width != 0 || (num_groups % num_virtual_cols) != 0) {
        num_virtual_cols -= 1;
    }

    uint32_t num_actual_cols = (grid_size.x / num_virtual_cols) * num_virtual_cols;
    uint32_t num_actual_rows = grid_size.y;
    uint32_t num_virtual_rows = (grid_size.x / num_virtual_cols) * num_actual_rows;
    uint32_t num_cores = num_actual_cols * num_actual_rows;
    const bool row_wise = false;
    auto all_cores = tt::tt_metal::num_cores_to_corerangeset(num_cores, grid_size, row_wise);

    TT_FATAL(
        Ht >= num_virtual_rows,
        "Height in tiles (Ht={}) must be >= num_virtual_rows ({}). "
        "The core grid (x={}, y={}) is too large for the input spatial dimensions (H={}). "
        "Use a smaller core_grid or increase the input spatial size.",
        Ht,
        num_virtual_rows,
        grid_size.x,
        grid_size.y,
        H);
    TT_FATAL(
        Ht % num_virtual_rows == 0,
        "Height in tiles (Ht={}) must be divisible by num_virtual_rows ({}). "
        "Remainder tiles would be silently dropped, producing incorrect results. "
        "core_grid=({},{}), num_virtual_cols={}, rows_per_y={}.",
        Ht,
        num_virtual_rows,
        grid_size.x,
        grid_size.y,
        num_virtual_cols,
        grid_size.x / num_virtual_cols);

    uint32_t per_core_Mt_group_1 = Ht / num_virtual_rows;
    uint32_t per_core_M_group_1 = per_core_Mt_group_1 * tile_height;
    uint32_t per_core_N = W / num_virtual_cols;
    uint32_t per_core_Nt = (per_core_N + tile_width - 1) / tile_width;
    uint32_t num_channels_per_group = W / num_groups;
    uint32_t num_channels_per_group_mod_tile_w =
        num_channels_per_group % tile_width == 0 ? tile_width : num_channels_per_group % tile_width;
    uint32_t num_shards_r = H / per_core_M_group_1;
    uint32_t num_cores_per_batch = num_batches > num_shards_r ? 1 : num_shards_r / num_batches;
    uint32_t num_shards_c = W / per_core_N;
    uint32_t num_cores_per_group = num_groups > num_shards_c ? 1 : num_shards_c / num_groups;
    uint32_t num_batches_per_core_group_1 = num_batches > num_shards_r ? num_batches / num_shards_r : 1;
    uint32_t num_groups_per_core = num_groups > num_shards_c ? num_groups / num_shards_c : 1;

    TT_FATAL(
        (!use_welford) || (num_groups_per_core <= 16),
        "num_groups_per_core ({}) must be <= 16 when use_welfords is true.",
        num_groups_per_core);

    // -1 sentinel from GroupNormMultiCoreProgramConfig means "auto select":
    // pick num_out_blocks from a simple input-size / grid-size heuristic, rounded
    // up to the next power of two and capped at MAX_HEURISTIC_NUM_OUT_BLOCKS.
    // Any other value is taken as an explicit user choice and validated below.
    if (num_out_blocks == static_cast<uint32_t>(-1)) {
        const uint32_t HEURISTIC_BLOCK_SIZE_BASE = 256 * 256;
        const uint32_t MAX_HEURISTIC_NUM_OUT_BLOCKS = 256;
        uint32_t heuristic_num_out_blocks =
            (shape[1] * shape[2] * shape[3]) / (HEURISTIC_BLOCK_SIZE_BASE * (num_virtual_cols * num_virtual_rows));
        heuristic_num_out_blocks = heuristic_num_out_blocks ? heuristic_num_out_blocks : 1;
        num_out_blocks = 1;
        while (num_out_blocks < heuristic_num_out_blocks && num_out_blocks < MAX_HEURISTIC_NUM_OUT_BLOCKS) {
            num_out_blocks <<= 1;
        }
    }

    TT_FATAL(
        num_batches_per_core_group_1 > 0,
        "num_batches_per_core_group_1 must be > 0 (got 0). This indicates an internal grid sizing error.");
    uint32_t num_rows_per_batch_per_core_group_1 = per_core_M_group_1 / num_batches_per_core_group_1;
    auto [block_wt, num_groups_per_reset] = find_max_tile_span(per_core_N, num_channels_per_group);
    uint32_t block_ht_group_1 = per_core_Mt_group_1 / num_batches_per_core_group_1;
    uint32_t subblock_wt = get_max_subblock(block_wt, 8);
    uint32_t num_subblocks_w = block_wt / subblock_wt;
    bool block_wt_last = (per_core_Nt + num_groups_per_core - 1) / num_groups_per_core;

    TT_FATAL(
        block_ht_group_1 > 0,
        "block_h (tile height per core per batch) is 0. The core grid is too large for the input spatial dimensions. "
        "per_core_Mt={}, num_batches_per_core={}, grid=({},{}), Ht={}.",
        per_core_Mt_group_1,
        num_batches_per_core_group_1,
        grid_size.x,
        grid_size.y,
        Ht);
    TT_FATAL(
        num_out_blocks > 0 && num_out_blocks <= block_ht_group_1,
        "num_out_blocks ({}) must be in [1, block_h ({})]. "
        "Reduce num_out_blocks or increase input spatial dimensions.",
        num_out_blocks,
        block_ht_group_1);

    bool equal_batches_per_core = true;
    uint32_t last_row_with_extra_batch = 0;
    if (num_batches >= num_shards_r) {
        last_row_with_extra_batch = (num_batches % num_shards_r);
        equal_batches_per_core = (last_row_with_extra_batch == 0);
        if (!equal_batches_per_core) {
            last_row_with_extra_batch--;
        }
    }

    uint32_t per_core_N_bytes_padded = tt::round_up(per_core_N * datum_size_bytes, output.buffer()->alignment());
    bool reader_repack_output = (per_core_N % tile_width) != 0;
    bool tilize_in = a.layout() == Layout::ROW_MAJOR;
    bool untilize_out = output.layout() == Layout::ROW_MAJOR;

    TT_FATAL(num_channels_per_group > 0, "num_channels_per_group must be > 0 (W={}, num_groups={})", W, num_groups);
    TT_FATAL(
        num_rows_per_batch_per_core_group_1 > 0,
        "num_rows_per_batch_per_core_group_1 must be > 0 (per_core_M={}, num_batches_per_core={})",
        per_core_M_group_1,
        num_batches_per_core_group_1);
    TT_FATAL(
        num_cores_per_batch > 0 && num_cores_per_group > 0,
        "num_cores_per_batch ({}) and num_cores_per_group ({}) must both be > 0",
        num_cores_per_batch,
        num_cores_per_group);
    TT_FATAL(
        per_core_N % num_channels_per_group == 0,
        "per_core_N ({}) must be divisible by num_channels_per_group ({})",
        per_core_N,
        num_channels_per_group);
    TT_FATAL(per_core_M_group_1 % tile_height == 0, "per_core_M must be divisible by TILE_HEIGHT");
    TT_FATAL(W % num_groups == 0, "Tensor W ({}) must be divisible by num_groups ({})", W, num_groups);
    TT_FATAL(W % per_core_N == 0, "W dim ({}) must be divisible by per_core_N ({})", W, per_core_N);

    if (input_mask.has_value()) {
        TT_FATAL(
            input_mask.value().padded_shape()[3] == block_wt * tile_width,
            "input mask width ({}) must have the same width as block_wt * TILE_WIDTH ({})",
            input_mask.value().padded_shape()[3],
            block_wt * tile_width);
    }

    auto in0_dram_addr = a.buffer()->address();
    auto out_dram_addr = output.buffer()->address();
    auto gamma_dram_addr = gamma.has_value() ? gamma.value().buffer()->address() : 0;
    auto beta_dram_addr = beta.has_value() ? beta.value().buffer()->address() : 0;
    auto input_mask_dram_addr = input_mask.has_value() ? input_mask.value().buffer()->address() : 0;

    uint32_t in0_block_tiles_group_1 = block_ht_group_1 / num_out_blocks * block_wt;
    uint32_t in0_CB_size_group_1 = in0_block_tiles_group_1 * in_single_tile_size;
    uint32_t in_CB_size_group_1 = in0_block_tiles_group_1 * in_single_tile_size;
    uint32_t in2_CB_size = single_tile_size;
    uint32_t in3_CB_size = single_tile_size;
    uint32_t gamma_beta_num_cols_tile_per_core = per_core_Nt;
    uint32_t in5_CB_size = gamma_beta_num_cols_tile_per_core * gamma_beta_single_tile_size;
    uint32_t in6_CB_size = gamma_beta_num_cols_tile_per_core * gamma_beta_single_tile_size;
    uint32_t input_mask_num_tiles_per_core = block_wt * num_groups_per_core;
    uint32_t in_mask_CB_size = use_welford ? input_mask.value().physical_volume() * input_mask.value().element_size()
                                           : block_wt * in_mask_single_tile_size * 2;
    uint32_t repack_CB_size = per_core_Nt * in_single_tile_size * 2;
    uint32_t interm_block_tiles_group_1 = in0_block_tiles_group_1;
    uint32_t x_CB_size_group_1 = interm_block_tiles_group_1 * single_tile_size;
    uint32_t xmm_CB_size_group_1 = interm_block_tiles_group_1 * single_tile_size;
    uint32_t ex_partial_CB_size = single_tile_size * (use_welford ? 2 : 1);
    uint32_t ex2_partial_CB_size = single_tile_size;
    uint32_t ex_global_CB_size = ex_partial_CB_size * (use_welford ? num_groups_per_core : 1);
    uint32_t ex2_global_CB_size = ex2_partial_CB_size;
    uint32_t xmm2_CB_size_group_1 = interm_block_tiles_group_1 * single_tile_size;
    uint32_t xmm3_CB_size_group_1 = interm_block_tiles_group_1 * single_tile_size;
    uint32_t ex2pe_CB_size = use_welford ? single_tile_size * num_groups_per_core : ex_partial_CB_size;
    uint32_t reciprocal_CB_size = reciprocals.has_value() ? reciprocals.value().buffer()->aligned_size_per_bank() : 0;
    uint32_t out_CB_size_group_1 = in0_block_tiles_group_1 * out_single_tile_size;

    if (use_welford) {
        x_CB_size_group_1 = single_tile_size * 1;
        xmm_CB_size_group_1 = single_tile_size * 3;
    }

    std::vector<CoreCoord> core_coords = grid_to_cores(num_cores, num_actual_cols, num_actual_rows, row_wise);
    std::vector<CoreCoord> virtual_core_coords = grid_to_cores(num_cores, num_virtual_cols, num_virtual_rows, row_wise);
    std::set<CoreRange> all_cores_group_1_core_ranges;
    for (size_t i = 0; i < num_cores; ++i) {
        all_cores_group_1_core_ranges.insert(CoreRange(core_coords[i]));
    }
    CoreRangeSet all_cores_group_1 = CoreRangeSet(all_cores_group_1_core_ranges);

    std::set<CoreRange> mcast_sender_core_ranges_group_1;
    std::set<CoreRange> mcast_sender_core_ranges_all;
    std::set<CoreRange> mcast_receiver_core_ranges_group_1;
    std::set<CoreRange> mcast_receiver_core_ranges_all;
    uint32_t core_index_offset = 0;
    uint32_t sender_groups_count =
        equal_batches_per_core ? (num_batches / num_batches_per_core_group_1) : num_virtual_rows;
    for (uint32_t i = 0; i < sender_groups_count; ++i) {
        uint32_t core_index = core_index_offset;
        for (uint32_t j = 0; j < num_groups / num_groups_per_core; ++j) {
            mcast_sender_core_ranges_all.insert(CoreRange(core_coords[core_index]));
            mcast_sender_core_ranges_group_1.insert(CoreRange(core_coords[core_index]));
            core_index += num_virtual_rows;
        }
        core_index_offset += num_cores_per_batch;
    }
    for (size_t i = 0; i < num_cores; ++i) {
        if (!mcast_sender_core_ranges_all.contains(CoreRange(core_coords[i]))) {
            mcast_receiver_core_ranges_all.insert(CoreRange(core_coords[i]));
            mcast_receiver_core_ranges_group_1.insert(CoreRange(core_coords[i]));
        }
    }
    CoreRangeSet mcast_sender_cores_group_1 = CoreRangeSet(mcast_sender_core_ranges_group_1);
    CoreRangeSet mcast_receiver_cores_group_1 = CoreRangeSet(mcast_receiver_core_ranges_group_1);

    std::vector<std::vector<CoreCoord>> mcast_groups;
    std::vector<std::vector<CoreCoord>> mcast_virtual_groups;
    int group_index = -1;
    for (size_t i = 0; i < core_coords.size(); ++i) {
        if (mcast_sender_core_ranges_all.contains(CoreRange(core_coords[i]))) {
            group_index += 1;
        }
        if (group_index >= static_cast<int>(mcast_groups.size())) {
            mcast_groups.push_back(std::vector<CoreCoord>());
            mcast_virtual_groups.push_back(std::vector<CoreCoord>());
        }
        mcast_groups[group_index].push_back(core_coords[i]);
        mcast_virtual_groups[group_index].push_back(virtual_core_coords[i]);
    }

    uint32_t num_cores_per_mcast_group = mcast_groups[0].size();

    // ---- Build ProgramDescriptor ----
    ProgramDescriptor desc;

    // Semaphores - sender (id=0), receiver (id=1)
    constexpr uint32_t reduce_sender_semaphore_id = 0;
    constexpr uint32_t reduce_receiver_semaphore_id = 1;
    desc.semaphores.push_back(SemaphoreDescriptor{
        .id = reduce_sender_semaphore_id,
        .core_type = tt::CoreType::WORKER,
        .core_ranges = all_cores,
        .initial_value = 0});
    desc.semaphores.push_back(SemaphoreDescriptor{
        .id = reduce_receiver_semaphore_id,
        .core_type = tt::CoreType::WORKER,
        .core_ranges = all_cores,
        .initial_value = 0});

    std::map<std::string, std::string> reader_mcast_sender_defines;
    std::map<std::string, std::string> reader_mcast_receiver_defines;
    if (gamma.has_value()) {
        reader_mcast_sender_defines["FUSE_GAMMA"] = "1";
        reader_mcast_receiver_defines["FUSE_GAMMA"] = "1";
    }
    if (beta.has_value()) {
        reader_mcast_sender_defines["FUSE_BETA"] = "1";
        reader_mcast_receiver_defines["FUSE_BETA"] = "1";
    }
    if (reader_repack_output) {
        reader_mcast_sender_defines["READER_REPACK"] = "1";
        reader_mcast_receiver_defines["READER_REPACK"] = "1";
    }
    if (tilize_in) {
        reader_mcast_sender_defines["TILIZE_IN"] = "1";
        reader_mcast_receiver_defines["TILIZE_IN"] = "1";
    }
    if (untilize_out) {
        reader_mcast_sender_defines["UNTILIZE_OUT"] = "1";
        reader_mcast_receiver_defines["UNTILIZE_OUT"] = "1";
    }

    std::vector<uint32_t> reader_mcast_sender_compile_time_args_group_1 = {};
    std::unordered_map<std::string, uint32_t> reader_mcast_sender_named_compile_time_args = {
        {"reduce_receiver_semaphore_id", reduce_receiver_semaphore_id},
        {"reduce_sender_semaphore_id", reduce_sender_semaphore_id},
        {"num_cores_per_mcast_group", num_cores_per_mcast_group},
        {"num_batch_group", num_groups_per_core * num_batches_per_core_group_1},
        {"num_batches", num_batches_per_core_group_1},
        {"per_core_N", per_core_Nt},
        {"per_core_N_bytes", per_core_N_bytes_padded},
        {"per_core_N_bytes_with_stride", per_core_Nt * tile_width * datum_size_bytes},
        {"datum_size_bytes", datum_size_bytes},
        {"per_core_M", per_core_Mt_group_1},
        {"TILE_HEIGHT", tile_height},
        {"TILE_WIDTH", tile_width},
        {"block_h", block_ht_group_1},
        {"block_w", block_wt},
        {"block_hw", block_ht_group_1 * block_wt},
        {"num_cols_per_group", num_channels_per_group_mod_tile_w},
        {"num_tiles_per_batch", per_core_Mt_group_1 * Wt / num_batches_per_core_group_1},
        {"block_w_last", block_wt_last},
        {"GROUP_SIZE_IS_POWER_OF_2",
         static_cast<uint32_t>((num_channels_per_group_mod_tile_w & (num_channels_per_group_mod_tile_w - 1)) == 0)},
        {"GROUP_SIZE_SMALLER_THAN_TILE_W", static_cast<uint32_t>(num_channels_per_group < tile_width)},
        {"group_row_offset", num_channels_per_group - ((block_wt - 1) * tile_width)},
        {"num_out_blocks", num_out_blocks},
        {"num_channels_per_group", num_channels_per_group},
        {"num_rows_per_group", num_rows_per_batch_per_core_group_1},
    };

    tt::tt_metal::TensorAccessorArgs(a.buffer()).append_to(reader_mcast_sender_compile_time_args_group_1);
    tt::tt_metal::TensorAccessorArgs(output.buffer()).append_to(reader_mcast_sender_compile_time_args_group_1);
    std::vector<uint32_t> reader_mcast_receiver_compile_time_args_group_1 = {};
    std::unordered_map<std::string, uint32_t> reader_mcast_receiver_named_compile_time_args = {
        {"reduce_receiver_semaphore_id", reduce_receiver_semaphore_id},
        {"reduce_sender_semaphore_id", reduce_sender_semaphore_id},
        {"num_batch_group", num_groups_per_core * num_batches_per_core_group_1},
        {"num_batches", num_batches_per_core_group_1},
        {"per_core_N", per_core_Nt},
        {"per_core_N_bytes", per_core_N_bytes_padded},
        {"per_core_N_bytes_with_stride", per_core_Nt * tile_width * datum_size_bytes},
        {"per_core_M", per_core_Mt_group_1},
        {"TILE_HEIGHT", tile_height},
        {"TILE_WIDTH", tile_width},
        {"block_h", block_ht_group_1},
        {"block_w", block_wt},
        {"block_hw", block_ht_group_1 * block_wt},
        {"num_cols_per_group", num_channels_per_group_mod_tile_w},
        {"num_tiles_per_batch", per_core_Mt_group_1 * Wt / num_batches_per_core_group_1},
        {"block_w_last", block_wt_last},
        {"GROUP_SIZE_IS_POWER_OF_2",
         static_cast<uint32_t>((num_channels_per_group_mod_tile_w & (num_channels_per_group_mod_tile_w - 1)) == 0)},
        {"GROUP_SIZE_SMALLER_THAN_TILE_W", static_cast<uint32_t>(num_channels_per_group < tile_width)},
        {"group_row_offset", num_channels_per_group - ((block_wt - 1) * tile_width)},
        {"num_out_blocks", num_out_blocks},
        {"num_channels_per_group", num_channels_per_group},
        {"num_rows_per_group", num_rows_per_batch_per_core_group_1},
    };

    tt::tt_metal::TensorAccessorArgs(a.buffer()).append_to(reader_mcast_receiver_compile_time_args_group_1);
    tt::tt_metal::TensorAccessorArgs(output.buffer()).append_to(reader_mcast_receiver_compile_time_args_group_1);
    tt::tt_metal::NOC writer_noc = tt::tt_metal::detail::preferred_noc_for_dram_write(device->arch());
    tt::tt_metal::NOC reader_noc = tt::tt_metal::detail::preferred_noc_for_dram_read(device->arch());

    std::string reader_sender_kernel_path =
        (use_welford ? "ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/dataflow/"
                       "welford_reader_mcast_sender_unary_gn.cpp"
                     : "ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/dataflow/"
                       "reader_mcast_sender_unary_gn.cpp");
    std::string reader_receiver_kernel_path =
        (use_welford ? "ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/dataflow/"
                       "welford_reader_mcast_receiver_unary_gn.cpp"
                     : "ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/dataflow/"
                       "reader_mcast_receiver_unary_gn.cpp");

    KernelDescriptor reader_mcast_sender_desc;
    reader_mcast_sender_desc.kernel_source = reader_sender_kernel_path;
    reader_mcast_sender_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_mcast_sender_desc.core_ranges = mcast_sender_cores_group_1;
    reader_mcast_sender_desc.compile_time_args = reader_mcast_sender_compile_time_args_group_1;
    reader_mcast_sender_desc.named_compile_time_args = to_named_args_mcast(reader_mcast_sender_named_compile_time_args);
    reader_mcast_sender_desc.defines =
        KernelDescriptor::Defines(reader_mcast_sender_defines.begin(), reader_mcast_sender_defines.end());
    reader_mcast_sender_desc.config = DataMovementConfigDescriptor{
        .processor = DataMovementProcessor::RISCV_0,
        .noc = reader_noc,
    };

    KernelDescriptor reader_mcast_receiver_desc;
    reader_mcast_receiver_desc.kernel_source = reader_receiver_kernel_path;
    reader_mcast_receiver_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_mcast_receiver_desc.core_ranges = mcast_receiver_cores_group_1;
    reader_mcast_receiver_desc.compile_time_args = reader_mcast_receiver_compile_time_args_group_1;
    reader_mcast_receiver_desc.named_compile_time_args =
        to_named_args_mcast(reader_mcast_receiver_named_compile_time_args);
    reader_mcast_receiver_desc.defines =
        KernelDescriptor::Defines(reader_mcast_receiver_defines.begin(), reader_mcast_receiver_defines.end());
    reader_mcast_receiver_desc.config = DataMovementConfigDescriptor{
        .processor = DataMovementProcessor::RISCV_0,
        .noc = reader_noc,
    };

    std::vector<uint32_t> writer_mcast_sender_compile_time_args_group_1 = {};
    std::unordered_map<std::string, uint32_t> writer_named_compile_time_args_group_1 = {
        {"is_mcast_sender", 1},
        {"fuse_gamma", static_cast<uint32_t>(gamma.has_value())},
        {"fuse_beta", static_cast<uint32_t>(beta.has_value())},
        {"num_cols_tile_gamma_beta", gamma_beta_num_cols_tile_per_core},
        {"per_core_M", per_core_Mt_group_1},
        {"per_core_N", per_core_Nt},
        {"per_core_N_bytes", per_core_N * datum_size_bytes},
        {"per_core_N_bytes_with_stride", per_core_Nt * tile_width * datum_size_bytes},
        {"num_groups_per_core", num_groups_per_core},
        {"num_batches_per_core", num_batches_per_core_group_1},
        {"num_cols_per_group", num_channels_per_group_mod_tile_w},
        {"num_tiles_per_batch", per_core_Mt_group_1 * Wt / num_batches_per_core_group_1},
        {"block_w_last", block_wt_last},
        {"GROUP_SIZE_IS_POWER_OF_2",
         static_cast<uint32_t>((num_channels_per_group_mod_tile_w & (num_channels_per_group_mod_tile_w - 1)) == 0)},
        {"GROUP_SIZE_SMALLER_THAN_TILE_W", static_cast<uint32_t>(num_channels_per_group < tile_width)},
        {"group_row_offset", num_channels_per_group - ((block_wt - 1) * tile_width)},
        {"num_out_blocks", num_out_blocks},
        {"block_h", block_ht_group_1},
        {"block_w", block_wt},
        {"block_hw", block_ht_group_1 * block_wt},
        {"groupnorm_mode", groupnorm_mode},
        {"TILE_WIDTH", tile_width},
        {"TILE_HW", tile_hw},
        {"reduce_factor_w", num_rows_per_batch_per_core_group_1 * num_channels_per_group},
        {"reduce_factor_c", num_cores_per_batch * num_cores_per_group},
    };

    if (gamma.has_value() && gamma.value().layout() == Layout::ROW_MAJOR) {
        auto gamma_stick_size = gamma.value().padded_shape()[3] * gamma.value().element_size();
        writer_named_compile_time_args_group_1["page_size"] = gamma_stick_size;
    } else if (beta.has_value() && beta.value().layout() == Layout::ROW_MAJOR) {
        auto beta_stick_size = beta.value().padded_shape()[3] * beta.value().element_size();
        writer_named_compile_time_args_group_1["page_size"] = beta_stick_size;
    } else {
        writer_named_compile_time_args_group_1["page_size"] = tile_hw * datum_size_bytes;
    }

    tt::tt_metal::TensorAccessorArgs(output.buffer()).append_to(writer_mcast_sender_compile_time_args_group_1);
    tt::tt_metal::TensorAccessorArgs(gamma.has_value() ? gamma.value().buffer() : nullptr)
        .append_to(writer_mcast_sender_compile_time_args_group_1);
    tt::tt_metal::TensorAccessorArgs(beta.has_value() ? beta.value().buffer() : nullptr)
        .append_to(writer_mcast_sender_compile_time_args_group_1);
    tt::tt_metal::TensorAccessorArgs(input_mask.has_value() ? input_mask.value().buffer() : nullptr)
        .append_to(writer_mcast_sender_compile_time_args_group_1);

    std::string writer_kernel =
        (use_welford ? "ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/dataflow/"
                       "welford_writer_unary_gn_rm_gb.cpp"
                     : "ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/dataflow/"
                       "writer_unary_gn_rm_gb.cpp");

    KernelDescriptor writer_desc;
    writer_desc.kernel_source = writer_kernel;
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores_group_1;
    writer_desc.compile_time_args = writer_mcast_sender_compile_time_args_group_1;
    writer_desc.named_compile_time_args = to_named_args_mcast(writer_named_compile_time_args_group_1);
    writer_desc.config = DataMovementConfigDescriptor{
        .processor = DataMovementProcessor::RISCV_1,
        .noc = writer_noc,
    };

    std::map<std::string, std::string> eltwise_binary_defines;
    if (reader_repack_output) {
        eltwise_binary_defines["READER_REPACK"] = "1";
    }
    if (tilize_in) {
        eltwise_binary_defines["TILIZE_IN"] = "1";
    }
    if (untilize_out) {
        eltwise_binary_defines["UNTILIZE_OUT"] = "1";
    }

    std::vector<uint32_t> mcast_sender_compute_compile_time_args_group_1 = {};
    std::unordered_map<std::string, uint32_t> mcast_sender_compute_named_compile_time_args = {
        {"is_mcast_sender", 1},
        {"do_gamma", static_cast<uint32_t>(gamma.has_value())},
        {"do_beta", static_cast<uint32_t>(beta.has_value())},
        {"num_cores_per_mcast_group", num_cores_per_mcast_group},
        {"batch", num_batches_per_core_group_1},
        {"group", num_groups_per_core},
        {"block_h", block_ht_group_1},
        {"block_w", block_wt},
        {"block_hw", block_ht_group_1 * block_wt},
        {"subblock_w", subblock_wt},
        {"num_subblocks_w", num_subblocks_w},
        {"per_core_M", per_core_Mt_group_1},
        {"per_core_N", per_core_Nt},
        {"per_core_MN", per_core_Mt_group_1 * per_core_Nt},
        {"per_core_N_tile_bytes", per_core_Nt * tile_hw * datum_size_bytes},
        {"num_groups_per_reset", num_groups_per_reset},
        {"single_tile_size_bytes", single_tile_size},
        {"num_tiles_per_batch", per_core_Mt_group_1 * Wt / num_batches_per_core_group_1},
        {"num_tiles_input_mask", num_groups_per_core * block_wt},
        {"num_cols_per_group", num_channels_per_group_mod_tile_w},
        {"block_w_last", block_wt_last},
        {"GROUP_SIZE_IS_POWER_OF_2",
         static_cast<uint32_t>((num_channels_per_group_mod_tile_w & (num_channels_per_group_mod_tile_w - 1)) == 0)},
        {"GROUP_SIZE_SMALLER_THAN_TILE_W", static_cast<uint32_t>(num_channels_per_group < tile_width)},
        {"group_row_offset", num_channels_per_group - ((block_wt - 1) * tile_width)},
        {"num_out_blocks", num_out_blocks},
        {"num_channels_per_group", num_channels_per_group},
        {"num_rows_per_group", num_rows_per_batch_per_core_group_1},
        {"reciprocal_size", num_reciprocals},
        {"TILE_WIDTH", tile_width},
    };

    std::vector<uint32_t> mcast_receiver_compute_compile_time_args_group_1 = {};
    std::unordered_map<std::string, uint32_t> mcast_receiver_compute_named_compile_time_args = {
        {"is_mcast_sender", 0},
        {"do_gamma", static_cast<uint32_t>(gamma.has_value())},
        {"do_beta", static_cast<uint32_t>(beta.has_value())},
        {"num_cores_per_mcast_group", num_cores_per_mcast_group},
        {"batch", num_batches_per_core_group_1},
        {"group", num_groups_per_core},
        {"block_h", block_ht_group_1},
        {"block_w", block_wt},
        {"block_hw", block_ht_group_1 * block_wt},
        {"subblock_w", subblock_wt},
        {"num_subblocks_w", num_subblocks_w},
        {"per_core_M", per_core_Mt_group_1},
        {"per_core_N", per_core_Nt},
        {"per_core_MN", per_core_Mt_group_1 * per_core_Nt},
        {"per_core_N_tile_bytes", per_core_Nt * tile_hw * datum_size_bytes},
        {"num_groups_per_reset", num_groups_per_reset},
        {"single_tile_size_bytes", single_tile_size},
        {"num_tiles_per_batch", per_core_Mt_group_1 * Wt / num_batches_per_core_group_1},
        {"num_tiles_input_mask", num_groups_per_core * block_wt},
        {"num_cols_per_group", num_channels_per_group_mod_tile_w},
        {"block_w_last", block_wt_last},
        {"GROUP_SIZE_IS_POWER_OF_2",
         static_cast<uint32_t>((num_channels_per_group_mod_tile_w & (num_channels_per_group_mod_tile_w - 1)) == 0)},
        {"GROUP_SIZE_SMALLER_THAN_TILE_W", static_cast<uint32_t>(num_channels_per_group < tile_width)},
        {"group_row_offset", num_channels_per_group - ((block_wt - 1) * tile_width)},
        {"num_out_blocks", num_out_blocks},
        {"num_channels_per_group", num_channels_per_group},
        {"num_rows_per_group", num_rows_per_batch_per_core_group_1},
        {"reciprocal_size", num_reciprocals},
        {"TILE_WIDTH", tile_width},
    };

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);
    eltwise_binary_defines["FP32_DEST_ACC"] = fp32_dest_acc_en ? "true" : "false";

    std::string compute_kernel_path =
        (use_welford ? "ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/compute/welford_groupnorm.cpp"
                     : "ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/compute/groupnorm.cpp");

    // Float32 input requires fp32 dest accumulation; otherwise the unpacker would silently
    // downcast through SrcA to TF32 / Float16_b (~10 mantissa bits).
    TT_FATAL(
        !(use_welford && in_data_format == tt::DataFormat::Float32 && !fp32_dest_acc_en),
        "group_norm welford with Float32 input requires fp32_dest_acc_en=true in the compute "
        "kernel config; otherwise precision is silently lost in the unpacker format conversion.");

    // For Float32 input on the Welford path, force unpack-to-dest in fp32 mode so the unpacker
    // writes full fp32 to DEST instead of routing through SrcA (which downcasts to TF32 = 10
    // mantissa bits). Without this the Welford recurrence sees TF32-truncated inputs
    // and catastrophically loses precision when |mean| >> std.
    std::vector<tt::tt_metal::UnpackToDestMode> unpack_to_dest_mode(
        NUM_CIRCULAR_BUFFERS, tt::tt_metal::UnpackToDestMode::Default);
    if (use_welford && fp32_dest_acc_en && in_data_format == tt::DataFormat::Float32) {
        unpack_to_dest_mode[static_cast<uint32_t>(tt::CBIndex::c_0)] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
    }

    KernelDescriptor compute_sender_desc;
    compute_sender_desc.kernel_source = compute_kernel_path;
    compute_sender_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_sender_desc.core_ranges = mcast_sender_cores_group_1;
    compute_sender_desc.compile_time_args = mcast_sender_compute_compile_time_args_group_1;
    compute_sender_desc.named_compile_time_args = to_named_args_mcast(mcast_sender_compute_named_compile_time_args);
    compute_sender_desc.defines =
        KernelDescriptor::Defines(eltwise_binary_defines.begin(), eltwise_binary_defines.end());
    compute_sender_desc.config = ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .dst_full_sync_en = dst_full_sync_en,
        .unpack_to_dest_mode = unpack_to_dest_mode,
        .math_approx_mode = math_approx_mode,
    };

    KernelDescriptor compute_receiver_desc;
    compute_receiver_desc.kernel_source = compute_kernel_path;
    compute_receiver_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_receiver_desc.core_ranges = mcast_receiver_cores_group_1;
    compute_receiver_desc.compile_time_args = mcast_receiver_compute_compile_time_args_group_1;
    compute_receiver_desc.named_compile_time_args = to_named_args_mcast(mcast_receiver_compute_named_compile_time_args);
    compute_receiver_desc.defines =
        KernelDescriptor::Defines(eltwise_binary_defines.begin(), eltwise_binary_defines.end());
    compute_receiver_desc.config = ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .dst_full_sync_en = dst_full_sync_en,
        .unpack_to_dest_mode = unpack_to_dest_mode,
        .math_approx_mode = math_approx_mode,
    };

    // Create circular buffers
    constexpr uint32_t in0_cb_index = tt::CBIndex::c_0;
    constexpr uint32_t output_cb_index = tt::CBIndex::c_16;
    desc.cbs.push_back(CBDescriptor{
        .total_size = in0_CB_size_group_1,
        .core_ranges = all_cores_group_1,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(in0_cb_index),
            .data_format = in_data_format,
            .page_size = in_single_tile_size,
        }}},
    });
    desc.cbs.push_back(CBDescriptor{
        .total_size = out_CB_size_group_1,
        .core_ranges = all_cores_group_1,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(output_cb_index),
            .data_format = out_data_format,
            .page_size = out_single_tile_size,
        }}},
    });

    constexpr uint32_t in_cb_index = tt::CBIndex::c_29;
    desc.cbs.push_back(CBDescriptor{
        .total_size = in_CB_size_group_1,
        .core_ranges = all_cores_group_1,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(in_cb_index),
            .data_format = in_data_format,
            .page_size = in_single_tile_size,
        }}},
    });

    if (untilize_out) {
        constexpr uint32_t out_cb_index = tt::CBIndex::c_30;
        desc.cbs.push_back(CBDescriptor{
            .total_size = in_CB_size_group_1,
            .core_ranges = all_cores_group_1,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(out_cb_index),
                .data_format = in_data_format,
                .page_size = in_single_tile_size,
            }}},
        });
    }

    constexpr uint32_t in2_cb_index = tt::CBIndex::c_2;
    desc.cbs.push_back(CBDescriptor{
        .total_size = in2_CB_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(in2_cb_index),
            .data_format = cb_data_format,
            .page_size = single_tile_size,
        }}},
    });

    constexpr uint32_t in3_cb_index = tt::CBIndex::c_3;
    desc.cbs.push_back(CBDescriptor{
        .total_size = in3_CB_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(in3_cb_index),
            .data_format = cb_data_format,
            .page_size = single_tile_size,
        }}},
    });

    constexpr uint32_t in4_cb_index = tt::CBIndex::c_4;
    desc.cbs.push_back(CBDescriptor{
        .total_size = in2_CB_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(in4_cb_index),
            .data_format = cb_data_format,
            .page_size = single_tile_size,
        }}},
    });

    if (gamma.has_value()) {
        constexpr uint32_t in5_cb_index = tt::CBIndex::c_5;
        desc.cbs.push_back(CBDescriptor{
            .total_size = in5_CB_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(in5_cb_index),
                .data_format = gamma_beta_cb_data_format,
                .page_size = gamma_beta_single_tile_size,
            }}},
        });
    }
    if (beta.has_value()) {
        constexpr uint32_t in6_cb_index = tt::CBIndex::c_6;
        desc.cbs.push_back(CBDescriptor{
            .total_size = in6_CB_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(in6_cb_index),
                .data_format = gamma_beta_cb_data_format,
                .page_size = gamma_beta_single_tile_size,
            }}},
        });
    }
    if (input_mask.has_value()) {
        constexpr uint32_t in_mask_cb_index = tt::CBIndex::c_28;
        desc.cbs.push_back(CBDescriptor{
            .total_size = in_mask_CB_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(in_mask_cb_index),
                .data_format = in_mask_cb_data_format,
                .page_size = in_mask_single_tile_size,
            }}},
        });
    }
    if (reader_repack_output) {
        constexpr uint32_t repack_cb_index = tt::CBIndex::c_26;
        constexpr uint32_t repack_out_cb_index = tt::CBIndex::c_31;
        desc.cbs.push_back(CBDescriptor{
            .total_size = repack_CB_size,
            .core_ranges = all_cores,
            .format_descriptors =
                {{CBFormatDescriptor{
                      .buffer_index = static_cast<uint8_t>(repack_cb_index),
                      .data_format = in_data_format,
                      .page_size = in_single_tile_size,
                  },
                  CBFormatDescriptor{
                      .buffer_index = static_cast<uint8_t>(repack_out_cb_index),
                      .data_format = in_data_format,
                      .page_size = in_single_tile_size,
                  }}},
        });
    }

    constexpr uint32_t x_cb_index = tt::CBIndex::c_24;
    desc.cbs.push_back(CBDescriptor{
        .total_size = x_CB_size_group_1,
        .core_ranges = all_cores_group_1,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(x_cb_index),
            .data_format = cb_data_format,
            .page_size = single_tile_size,
        }}},
    });

    constexpr uint32_t xmm_cb_index = tt::CBIndex::c_25;
    desc.cbs.push_back(CBDescriptor{
        .total_size = xmm_CB_size_group_1,
        .core_ranges = all_cores_group_1,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(xmm_cb_index),
            .data_format = cb_data_format,
            .page_size = single_tile_size,
        }}},
    });

    constexpr uint32_t xmm2_cb_index = tt::CBIndex::c_23;
    desc.cbs.push_back(CBDescriptor{
        .total_size = xmm2_CB_size_group_1,
        .core_ranges = all_cores_group_1,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(xmm2_cb_index),
            .data_format = cb_data_format,
            .page_size = single_tile_size,
        }}},
    });

    constexpr uint32_t xmm3_cb_index = tt::CBIndex::c_22;
    desc.cbs.push_back(CBDescriptor{
        .total_size = xmm3_CB_size_group_1,
        .core_ranges = all_cores_group_1,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(xmm3_cb_index),
            .data_format = cb_data_format,
            .page_size = single_tile_size,
        }}},
    });

    constexpr uint32_t ex_cb_partial_index = tt::CBIndex::c_8;
    desc.cbs.push_back(CBDescriptor{
        .total_size = ex_partial_CB_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(ex_cb_partial_index),
            .data_format = cb_data_format,
            .page_size = single_tile_size,
        }}},
    });

    if (!use_welford) {
        constexpr uint32_t ex2_cb_partial_index = tt::CBIndex::c_21;
        desc.cbs.push_back(CBDescriptor{
            .total_size = ex_partial_CB_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(ex2_cb_partial_index),
                .data_format = cb_data_format,
                .page_size = single_tile_size,
            }}},
        });
    }

    // cb_ex_external holds packed cb_ex_external_slot_pitch_bytes-sized partial-reduction
    // scalars gathered from every core in the mcast group, for every out_block. The
    // reader kernel (reader_mcast_sender_unary_gn) and compute kernel (groupnorm) both
    // reserve / wait-for cb_ex_external_tiles_required tiles at once, where
    //   cb_ex_external_tiles_required =
    //       ceil(num_out_blocks_padded * num_mcast_cores * cb_ex_external_slot_pitch_bytes / tile_size)
    // so the CB must be at least that large. Mirror the kernel's
    // num_out_blocks_padded calculation to get the exact count.
    // Note that Welford does not use cb_ex_external.
    if (!use_welford) {
        constexpr uint32_t ex_cb_external_index = tt::CBIndex::c_10;
        uint32_t num_out_blocks_padded = num_out_blocks;
        uint32_t out_block_h_normal = block_ht_group_1 / num_out_blocks;
        if (block_ht_group_1 % num_out_blocks != 0) {
            uint32_t residual = block_ht_group_1 - (num_out_blocks * out_block_h_normal);
            num_out_blocks_padded += (residual / out_block_h_normal + 1);
        }
        uint32_t cb_ex_external_tiles =
            (num_out_blocks_padded * num_cores_per_mcast_group * cb_ex_external_slot_pitch_bytes + single_tile_size -
             1) /
            single_tile_size;
        desc.cbs.push_back(CBDescriptor{
            .total_size = cb_ex_external_tiles * single_tile_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(ex_cb_external_index),
                .data_format = cb_data_format,
                .page_size = single_tile_size,
            }}},
        });
    }

    constexpr uint32_t ex_cb_index = tt::CBIndex::c_9;
    constexpr uint32_t ex_global_cb_index = tt::CBIndex::c_15;
    desc.cbs.push_back(CBDescriptor{
        .total_size = ex_global_CB_size,
        .core_ranges = all_cores,
        .format_descriptors =
            {{CBFormatDescriptor{
                  .buffer_index = static_cast<uint8_t>(ex_global_cb_index),
                  .data_format = cb_data_format,
                  .page_size = single_tile_size,
              },
              CBFormatDescriptor{
                  .buffer_index = static_cast<uint8_t>(ex_cb_index),
                  .data_format = cb_data_format,
                  .page_size = single_tile_size,
              }}},
    });

    if (!use_welford) {
        constexpr uint32_t ex2_cb_index = tt::CBIndex::c_13;
        constexpr uint32_t ex2_global_cb_index = tt::CBIndex::c_14;
        desc.cbs.push_back(CBDescriptor{
            .total_size = ex2_global_CB_size,
            .core_ranges = all_cores,
            .format_descriptors =
                {{CBFormatDescriptor{
                      .buffer_index = static_cast<uint8_t>(ex2_global_cb_index),
                      .data_format = cb_data_format,
                      .page_size = single_tile_size,
                  },
                  CBFormatDescriptor{
                      .buffer_index = static_cast<uint8_t>(ex2_cb_index),
                      .data_format = cb_data_format,
                      .page_size = single_tile_size,
                  }}},
        });
    }

    constexpr uint32_t cb_ex2pe_index = tt::CBIndex::c_27;
    desc.cbs.push_back(CBDescriptor{
        .total_size = ex2pe_CB_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_ex2pe_index),
            .data_format = cb_data_format,
            .page_size = single_tile_size,
        }}},
    });

    if (reciprocals.has_value()) {
        constexpr uint32_t cb_reciprocals = tt::CBIndex::c_18;
        desc.cbs.push_back(CBDescriptor{
            .total_size = reciprocal_CB_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(cb_reciprocals),
                .data_format = reciprocal_cb_data_format,
                .page_size = reciprocal_CB_size,
            }}},
            .buffer = reciprocals.value().buffer(),
        });
    }

    // Runtime Args
    uint32_t eps_u = std::bit_cast<uint32_t>(eps);

    for (size_t i = 0; i < mcast_groups.size(); ++i) {
        auto group = mcast_groups[i];
        const auto& virtual_group = mcast_virtual_groups[i];
        bool rectangle_grid = is_rectangle_grid(group);

        for (size_t j = 0; j < group.size(); ++j) {
            CoreCoord core = group[j];
            CoreCoord virtual_core = virtual_group[j];
            uint32_t in0_start_id = per_core_Mt_group_1 * Wt * virtual_core.y + per_core_Nt * virtual_core.x;
            uint32_t out_tile_start_id = per_core_Mt_group_1 * Wt * virtual_core.y + per_core_Nt * virtual_core.x;

            if (j == 0) {  // mcast sender
                std::vector<CoreCoord> mcast_group_first;
                std::vector<CoreCoord> mcast_group_mid(group);
                std::vector<CoreCoord> mcast_group_last;
                if (!rectangle_grid) {
                    split_and_form_rectangle_grids(group, mcast_group_first, mcast_group_mid, mcast_group_last);
                }

                CoreCoord mcast_start = device->worker_core_from_logical_core(mcast_group_mid.front());
                CoreCoord mcast_end = device->worker_core_from_logical_core(mcast_group_mid.back());

                if (reader_noc == NOC::NOC_1) {
                    std::swap(mcast_start, mcast_end);
                }
                std::vector<uint32_t> mcast_sender_args;
                mcast_sender_args.push_back(in0_dram_addr);
                mcast_sender_args.push_back(out_dram_addr);
                mcast_sender_args.push_back(in0_start_id);
                mcast_sender_args.push_back(out_tile_start_id);
                mcast_sender_args.push_back(Wt);
                mcast_sender_args.push_back(static_cast<uint32_t>(!mcast_group_first.empty()));
                mcast_sender_args.push_back(static_cast<uint32_t>(!mcast_group_last.empty()));
                mcast_sender_args.push_back(mcast_start.x);
                mcast_sender_args.push_back(mcast_start.y);
                mcast_sender_args.push_back(mcast_end.x);
                mcast_sender_args.push_back(mcast_end.y);
                if (!mcast_group_first.empty()) {
                    mcast_sender_args.push_back(mcast_group_mid.size());
                } else {
                    mcast_sender_args.push_back(mcast_group_mid.size() - 1);
                }

                if (!mcast_group_first.empty()) {
                    CoreCoord mcast_first_start = device->worker_core_from_logical_core(mcast_group_first.front());
                    CoreCoord mcast_first_end = device->worker_core_from_logical_core(mcast_group_first.back());

                    if (reader_noc == NOC::NOC_1) {
                        std::swap(mcast_start, mcast_end);
                    }
                    mcast_sender_args.push_back(mcast_first_start.x);
                    mcast_sender_args.push_back(mcast_first_start.y);
                    mcast_sender_args.push_back(mcast_first_end.x);
                    mcast_sender_args.push_back(mcast_first_end.y);
                    mcast_sender_args.push_back(mcast_group_first.size() - 1);
                }
                if (!mcast_group_last.empty()) {
                    CoreCoord mcast_last_start = device->worker_core_from_logical_core(mcast_group_last.front());
                    CoreCoord mcast_last_end = device->worker_core_from_logical_core(mcast_group_last.back());

                    if (reader_noc == NOC::NOC_1) {
                        std::swap(mcast_start, mcast_end);
                    }
                    mcast_sender_args.push_back(mcast_last_start.x);
                    mcast_sender_args.push_back(mcast_last_start.y);
                    mcast_sender_args.push_back(mcast_last_end.x);
                    mcast_sender_args.push_back(mcast_last_end.y);
                    mcast_sender_args.push_back(mcast_group_last.size());
                }

                std::vector<uint32_t> mcast_noc_xy;
                for (const auto& gcore : group) {
                    CoreCoord coord = device->worker_core_from_logical_core(gcore);
                    mcast_noc_xy.push_back(coord.x);
                }
                for (const auto& gcore : group) {
                    CoreCoord coord = device->worker_core_from_logical_core(gcore);
                    mcast_noc_xy.push_back(coord.y);
                }
                mcast_sender_args.insert(mcast_sender_args.end(), mcast_noc_xy.begin(), mcast_noc_xy.end());
                reader_mcast_sender_desc.runtime_args.emplace_back(core, std::move(mcast_sender_args));
            } else {  // mcast receiver
                // NOTE: do not pass Buffer* here. in0_start_id/out_tile_start_id/Wt/mcast
                // coords are per-core and shape-derived; using BufferBinding would skip
                // create_descriptor() on cache hits and leave those scalars stale when a
                // later call collides on the same cache entry with different shape/grid.
                reader_mcast_receiver_desc.runtime_args.emplace_back(
                    core,
                    tt::tt_metal::KernelDescriptor::CoreRuntimeArgs{
                        a.buffer()->address(),
                        output.buffer()->address(),
                        in0_start_id,
                        out_tile_start_id,
                        Wt,
                        static_cast<uint32_t>(device->worker_core_from_logical_core(group.front()).x),
                        static_cast<uint32_t>(device->worker_core_from_logical_core(group.front()).y)});
            }
        }
    }

    // writer
    uint32_t gamma_tile_start_id = 0;
    uint32_t beta_tile_start_id = 0;
    uint32_t input_mask_tile_start_id = 0;
    uint32_t curr_virtual_core_x = 0;
    for (size_t i = 0; i < core_coords.size(); ++i) {
        auto core = core_coords[i];
        auto virtual_core = virtual_core_coords[i];

        uint32_t out_tile_start_id = per_core_Mt_group_1 * Wt * virtual_core.y + per_core_Nt * virtual_core.x;

        if (virtual_core.x > curr_virtual_core_x) {
            curr_virtual_core_x++;
            if (gamma.has_value()) {
                gamma_tile_start_id = (gamma_tile_start_id + gamma_beta_num_cols_tile_per_core) %
                                      (gamma.value().physical_volume() / tile_width);
            }
            if (beta.has_value()) {
                beta_tile_start_id = (beta_tile_start_id + gamma_beta_num_cols_tile_per_core) %
                                     (beta.value().physical_volume() / tile_width);
            }
            if (input_mask.has_value()) {
                input_mask_tile_start_id = (input_mask_tile_start_id + input_mask_num_tiles_per_core) %
                                           (input_mask.value().physical_volume() / tile_hw);
            }
        }

        std::vector<uint32_t> writer_mcast_sender_args;
        writer_mcast_sender_args.push_back(eps_u);
        writer_mcast_sender_args.push_back(out_dram_addr);
        writer_mcast_sender_args.push_back(gamma_dram_addr);
        writer_mcast_sender_args.push_back(beta_dram_addr);
        writer_mcast_sender_args.push_back(input_mask_dram_addr);
        writer_mcast_sender_args.push_back(out_tile_start_id);
        writer_mcast_sender_args.push_back(gamma_tile_start_id);
        writer_mcast_sender_args.push_back(beta_tile_start_id);
        writer_mcast_sender_args.push_back(input_mask_tile_start_id);
        writer_mcast_sender_args.push_back(Wt);
        writer_desc.runtime_args.emplace_back(core, std::move(writer_mcast_sender_args));
    }

    desc.kernels.push_back(std::move(reader_mcast_sender_desc));
    desc.kernels.push_back(std::move(reader_mcast_receiver_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_sender_desc));
    desc.kernels.push_back(std::move(compute_receiver_desc));

    return desc;
}

}  // namespace ttnn::prim

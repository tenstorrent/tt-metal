// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "groupnorm_no_mcast_program_factory.hpp"
#include "groupnorm_program_utils.hpp"

#include <string>
#include <optional>

#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include "ttnn/operations/math.hpp"

using uint32_t = std::uint32_t;
using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::operations::normalization::group_norm {

GroupNormNoMcastProgramFactory::cached_program_t GroupNormNoMcastProgramFactory::create(
    const GroupNormParams& operation_attributes, const GroupNormInputs& tensor_args, Tensor& tensor_return_value) {
    const auto& a = tensor_args.input;
    const auto& gamma = tensor_args.gamma;
    const auto& beta = tensor_args.beta;
    const auto& input_mask = tensor_args.input_mask;
    const auto& reciprocals = tensor_args.reciprocals;
    auto& output = tensor_return_value;

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

    // Mode is 0 for legacy groupnorm, 1 for welford groupnorm, 2 for groupnorm with reciprocals
    uint32_t groupnorm_mode = static_cast<uint32_t>(
        reciprocals.has_value() ? GroupNormMode::WELFORD_RECIPROCALS
        : use_welford           ? GroupNormMode::WELFORD_NATIVE
                                : GroupNormMode::LEGACY);
    uint32_t num_reciprocals = reciprocals.has_value() ? reciprocals.value().shard_spec().value().numel() : 0;

    // convert data format
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
    uint32_t datum_size_bytes = 2;  // bfloat16

    TT_FATAL(
        out_data_format == in_data_format,
        "input: {} and output: {} must be the same data format",
        in_data_format,
        out_data_format);

    // tile sizes
    uint32_t in_single_tile_size = tt::tile_size(in_data_format);
    uint32_t single_tile_size = tt::tile_size(cb_data_format);
    uint32_t out_single_tile_size = tt::tile_size(out_data_format);
    uint32_t gamma_beta_single_tile_size = tt::tile_size(gamma_beta_cb_data_format);
    uint32_t in_mask_single_tile_size = tt::tile_size(in_mask_cb_data_format);

    IDevice* device = a.device();

    // tensor shape
    const auto& shape = a.padded_shape();
    uint32_t H = shape[1] * shape[2] * num_batches;
    uint32_t Ht = H / TILE_HEIGHT;
    uint32_t W = shape[3];
    uint32_t Wt = W / TILE_WIDTH;

    // Compute optimal core grid
    TT_FATAL(W % TILE_WIDTH == 0, "W (channels): {} must be divisible by {}", W, TILE_WIDTH);
    TT_FATAL(W % num_groups == 0, "W (channels): {} must be divisible by num_groups: {}", W, num_groups);
    uint32_t num_virtual_cols = std::min<uint32_t>(grid_size.x, num_groups);
    while ((W / num_virtual_cols) % TILE_WIDTH != 0 || (num_groups % num_virtual_cols) != 0) {
        num_virtual_cols -= 1;
    }

    uint32_t num_actual_cols =
        (grid_size.x / num_virtual_cols) * num_virtual_cols;  // Largest multiple of virtual cols < 8
    uint32_t num_actual_rows = grid_size.y;
    uint32_t num_virtual_rows = (grid_size.x / num_virtual_cols) * num_actual_rows;
    uint32_t num_cores = num_actual_cols * num_actual_rows;
    const bool row_wise = false;
    auto all_cores = tt::tt_metal::num_cores_to_corerangeset(num_cores, grid_size, row_wise);

    TT_FATAL(
        H >= num_virtual_rows,
        "Total size of a slice across channel dimension:({}) must be greater than or equal to num_virtual_rows: ({}). "
        "Reduce grid_size as needed",
        H,
        num_virtual_rows);

    uint32_t per_core_Mt_group_1 = Ht / num_virtual_rows;
    uint32_t per_core_M_group_1 = per_core_Mt_group_1 * TILE_HEIGHT;
    uint32_t per_core_Mt_group_2 = 0;
    uint32_t per_core_M_group_2 = 0;
    uint32_t per_core_N = W / num_virtual_cols;
    uint32_t per_core_Nt = (per_core_N + TILE_WIDTH - 1) / TILE_WIDTH;
    uint32_t num_channels_per_group = W / num_groups;
    uint32_t num_channels_per_group_mod_tile_w =
        num_channels_per_group % TILE_WIDTH == 0 ? TILE_WIDTH : num_channels_per_group % TILE_WIDTH;
    // split each batch into multiple cores
    uint32_t num_shards_r = H / per_core_M_group_1;
    uint32_t num_cores_per_batch = num_batches > num_shards_r ? 1 : num_shards_r / num_batches;
    uint32_t num_shards_c = W / per_core_N;
    uint32_t num_cores_per_group = num_groups > num_shards_c ? 1 : num_shards_c / num_groups;
    // each core contains multiple batches
    uint32_t num_batches_per_core_group_1 = num_batches > num_shards_r ? num_batches / num_shards_r : 1;
    uint32_t num_batches_per_core_group_2 = num_batches_per_core_group_1;  // need this to be non-zero even if unused
    uint32_t num_groups_per_core = num_groups > num_shards_c ? num_groups / num_shards_c : 1;

    TT_FATAL(
        (!use_welford) || (num_groups_per_core <= 16),
        "num_groups_per_core ({}) must be <= 16 when use_welfords is true. Increase the width of core grid to address "
        "this.",
        num_groups_per_core);

    // Compute num_out_blocks if not provided
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

    // subblock
    uint32_t num_rows_per_batch_per_core_group_1 = per_core_M_group_1 / num_batches_per_core_group_1;
    uint32_t num_rows_per_batch_per_core_group_2 = 0;
    auto [block_wt, num_groups_per_reset] = find_max_tile_span(per_core_N, num_channels_per_group);
    uint32_t block_ht_group_1 = per_core_Mt_group_1 / num_batches_per_core_group_1;
    uint32_t block_ht_group_2 = 0;
    uint32_t subblock_wt = get_max_subblock(block_wt, 8);
    uint32_t num_subblocks_w = block_wt / subblock_wt;
    bool block_wt_last = (per_core_Nt + num_groups_per_core - 1) / num_groups_per_core;

    // support for uneven batches across rows
    bool equal_batches_per_core = true;
    uint32_t last_row_with_extra_batch = 0;
    if (num_batches >= num_shards_r) {
        last_row_with_extra_batch = (num_batches % num_shards_r);
        equal_batches_per_core = (last_row_with_extra_batch == 0);
        if (!equal_batches_per_core) {
            last_row_with_extra_batch--;  // zero based index
        }
    }

    // Have first group (each row has 1 extra batch compared to second group), and second group
    if (!equal_batches_per_core) {
        num_batches_per_core_group_2 = num_batches / num_shards_r;
        num_batches_per_core_group_1 = num_batches_per_core_group_2 + 1;

        TT_FATAL(Ht % num_batches == 0, "Ht ({}) needs to be divisible by the number of batches ({})", Ht, num_batches);
        uint32_t per_batch_tiles = Ht / num_batches;
        per_core_Mt_group_1 = num_batches_per_core_group_1 * per_batch_tiles;
        per_core_Mt_group_2 = num_batches_per_core_group_2 * per_batch_tiles;
        per_core_M_group_1 = per_core_Mt_group_1 * TILE_HEIGHT;
        per_core_M_group_2 = per_core_Mt_group_2 * TILE_HEIGHT;

        num_rows_per_batch_per_core_group_1 = per_batch_tiles * TILE_HEIGHT;
        num_rows_per_batch_per_core_group_2 = per_batch_tiles * TILE_HEIGHT;

        block_ht_group_1 = per_batch_tiles;
        block_ht_group_2 = per_batch_tiles;
    }

    // shard shape per core
    uint32_t per_core_N_bytes_padded = tt::round_up(per_core_N * datum_size_bytes, output.buffer()->alignment());
    bool reader_repack_output = (per_core_N % TILE_WIDTH) != 0;
    bool tilize_in = a.layout() == Layout::ROW_MAJOR;
    bool untilize_out = output.layout() == Layout::ROW_MAJOR;

    TT_FATAL(
        per_core_N % num_channels_per_group == 0,
        "per_core_N ({}) must be divisible by num_channels_per_group ({})",
        per_core_N,
        num_channels_per_group);
    TT_FATAL(num_channels_per_group != 0, "num_channels_per_group should not equal 0");
    TT_FATAL(per_core_M_group_1 % TILE_HEIGHT == 0, "per_core_M: {} divides Tile Height", per_core_M_group_1);
    if (per_core_M_group_2 > 0) {
        TT_FATAL(per_core_M_group_2 % TILE_HEIGHT == 0, "per_core_M: {} divides Tile Height", per_core_M_group_2);
    }
    TT_FATAL(per_core_M_group_1 % TILE_HEIGHT == 0, "per_core_M must be divisible by TILE_HEIGHT");
    if (per_core_M_group_2 > 0) {
        TT_FATAL(per_core_M_group_2 % TILE_HEIGHT == 0, "per_core_M must be divisible by TILE_HEIGHT");
    }

    TT_FATAL(W % num_groups == 0, "Tensor W ({}) must be divisible by num_groups ({})", W, num_groups);
    TT_FATAL(W % per_core_N == 0, "W dim ({}) must be divisible by per_core_N ({})", W, per_core_N);
    if (num_batches < num_shards_r) {
        TT_FATAL(per_core_M_group_1 != 0, "per_core_M_group_1 should not equal 0");
        TT_FATAL(H % per_core_M_group_1 == 0, "H dim must be divisible by per_core_M");
        TT_FATAL(num_batches != 0, "num_batches should not equal 0");
        TT_FATAL(num_shards_r % num_batches == 0, "number of cores in a full column must be divisible by num_batches");
    }
    if (num_groups >= num_shards_c) {
        TT_FATAL(num_shards_c != 0, "num_shards_c should not equal 0");
        TT_FATAL(num_groups % num_shards_c == 0, "num_groups must be divisible by number of cores in a full row");
    } else {
        TT_FATAL(num_groups != 0, "num_group should not equal 0");
        TT_FATAL(num_shards_c % num_groups == 0, "number of cores in a full row must be divisible by num_groups");
    }

    log_debug(tt::LogOp, "num_cores: {}", num_cores);
    log_debug(tt::LogOp, "num_rows_per_batch_per_core_group 1: {}", num_rows_per_batch_per_core_group_1);
    log_debug(tt::LogOp, "num_rows_per_batch_per_core_group 2: {}", num_rows_per_batch_per_core_group_2);
    log_debug(tt::LogOp, "per_core_M_group_1: {}", per_core_M_group_1);
    log_debug(tt::LogOp, "per_core_M_group_2: {}", per_core_M_group_2);
    log_debug(tt::LogOp, "per_core_N: {}", per_core_N);
    log_debug(tt::LogOp, "W: {}", W);
    log_debug(tt::LogOp, "H: {}", H);
    log_debug(tt::LogOp, "num_channels_per_group: {}", num_channels_per_group);
    log_debug(tt::LogOp, "num_batches: {}", num_batches);
    log_debug(tt::LogOp, "num_groups: {}", num_groups);

    TT_FATAL(num_batches_per_core_group_1 != 0, "num_batches_per_core_group_1 should not equal 0");
    TT_FATAL(
        per_core_M_group_1 % num_batches_per_core_group_1 == 0,
        "per_core_M height must be divisible by per_core_batch");
    if (per_core_M_group_2 > 0) {
        TT_FATAL(num_batches_per_core_group_2 != 0, "num_batches_per_core_group_2 should not equal 0");
        TT_FATAL(
            per_core_M_group_2 % num_batches_per_core_group_2 == 0,
            "per_core_M height must be divisible by per_core_batch");
    }
    TT_FATAL(num_groups != 0, "num_groups should not equal 0");
    TT_FATAL(W % num_groups == 0, "tensor width must be divisible by num_groups ({})", num_groups);

    if (input_mask.has_value()) {
        TT_FATAL(
            input_mask.value().padded_shape()[3] == block_wt * TILE_WIDTH,
            "input mask width ({}) must have the same width as block_wt * TILE_WIDTH ({})",
            input_mask.value().padded_shape()[3],
            block_wt * TILE_WIDTH);
    }

    // get addr
    auto in0_dram_addr = a.buffer()->address();
    auto out_dram_addr = output.buffer()->address();
    auto gamma_dram_addr = gamma.has_value() ? gamma.value().buffer()->address() : 0;
    auto beta_dram_addr = beta.has_value() ? beta.value().buffer()->address() : 0;
    auto input_mask_dram_addr = input_mask.has_value() ? input_mask.value().buffer()->address() : 0;

    // Parameters Setup
    uint32_t in0_block_tiles_group_1 = block_ht_group_1 / num_out_blocks * block_wt;
    uint32_t in0_block_tiles_group_2 = 0;
    uint32_t in0_CB_size_group_1 = in0_block_tiles_group_1 * in_single_tile_size;
    uint32_t in0_CB_size_group_2 = 0;
    uint32_t in_CB_size_group_1 = in0_block_tiles_group_1 * in_single_tile_size;
    uint32_t in_CB_size_group_2 = 0;
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
    uint32_t interm_block_tiles_group_2 = 0;
    uint32_t x_CB_size_group_1 = interm_block_tiles_group_1 * single_tile_size;
    uint32_t x_CB_size_group_2 = 0;
    uint32_t xmm_CB_size_group_1 = interm_block_tiles_group_1 * single_tile_size;
    uint32_t xmm_CB_size_group_2 = 0;
    uint32_t ex_partial_CB_size = single_tile_size * (use_welford ? 2 : 1);
    uint32_t ex2_partial_CB_size = single_tile_size;
    uint32_t ex_global_CB_size = ex_partial_CB_size * (use_welford ? num_groups_per_core : 1);
    uint32_t ex2_global_CB_size = ex2_partial_CB_size;
    uint32_t xmm2_CB_size_group_1 = interm_block_tiles_group_1 * single_tile_size;
    uint32_t xmm2_CB_size_group_2 = 0;
    uint32_t xmm3_CB_size_group_1 = interm_block_tiles_group_1 * single_tile_size;
    uint32_t xmm3_CB_size_group_2 = 0;
    uint32_t ex2pe_CB_size = use_welford ? single_tile_size * num_groups_per_core : ex_partial_CB_size;
    uint32_t reciprocal_CB_size = reciprocals.has_value() ? reciprocals.value().buffer()->aligned_size_per_bank() : 0;
    uint32_t out_CB_size_group_1 = in0_block_tiles_group_1 * out_single_tile_size;
    uint32_t out_CB_size_group_2 = 0;

    if (!equal_batches_per_core) {
        in0_block_tiles_group_2 = block_ht_group_2 / num_out_blocks * block_wt;
        in0_CB_size_group_1 = in0_block_tiles_group_1 * in_single_tile_size;
        in0_CB_size_group_2 = in0_block_tiles_group_2 * in_single_tile_size;
        in_CB_size_group_1 = in0_block_tiles_group_1 * in_single_tile_size;
        in_CB_size_group_2 = in0_block_tiles_group_2 * in_single_tile_size;
        interm_block_tiles_group_1 = in0_block_tiles_group_1;
        interm_block_tiles_group_2 = in0_block_tiles_group_2;
        x_CB_size_group_1 = interm_block_tiles_group_1 * single_tile_size;
        x_CB_size_group_2 = interm_block_tiles_group_2 * single_tile_size;
        xmm_CB_size_group_1 = interm_block_tiles_group_1 * single_tile_size;
        xmm_CB_size_group_2 = interm_block_tiles_group_2 * single_tile_size;
        xmm2_CB_size_group_1 = interm_block_tiles_group_1 * single_tile_size;
        xmm2_CB_size_group_2 = interm_block_tiles_group_2 * single_tile_size;
        xmm3_CB_size_group_1 = interm_block_tiles_group_1 * single_tile_size;
        xmm3_CB_size_group_2 = interm_block_tiles_group_2 * single_tile_size;
        out_CB_size_group_1 = in0_block_tiles_group_1 * out_single_tile_size;
        out_CB_size_group_2 = in0_block_tiles_group_2 * out_single_tile_size;
    }

    if (use_welford) {
        x_CB_size_group_1 = single_tile_size * 1;
        x_CB_size_group_2 = single_tile_size * 1;
        xmm_CB_size_group_1 = single_tile_size * 3;
        xmm_CB_size_group_2 = single_tile_size * 3;
    }

    // Application Setup
    Program program = Program();

    std::vector<CoreCoord> core_coords = grid_to_cores(num_cores, num_actual_cols, num_actual_rows, row_wise);
    std::vector<CoreCoord> virtual_core_coords = grid_to_cores(num_cores, num_virtual_cols, num_virtual_rows, row_wise);
    std::set<CoreRange> all_cores_group_1_core_ranges;
    std::set<CoreRange> all_cores_group_2_core_ranges;
    for (size_t i = 0; i < num_cores; ++i) {
        CoreCoord virtual_core = virtual_core_coords[i];
        if (equal_batches_per_core || (virtual_core.y <= last_row_with_extra_batch)) {
            all_cores_group_1_core_ranges.insert(CoreRange(core_coords[i]));
        } else {
            all_cores_group_2_core_ranges.insert(CoreRange(core_coords[i]));
        }
    }
    CoreRangeSet all_cores_group_1 = CoreRangeSet(all_cores_group_1_core_ranges);
    CoreRangeSet all_cores_group_2 = CoreRangeSet(all_cores_group_2_core_ranges);

    std::set<CoreRange> mcast_sender_core_ranges_group_1;
    std::set<CoreRange> mcast_sender_core_ranges_group_2;
    std::set<CoreRange> mcast_sender_core_ranges_all;
    uint32_t core_index_offset = 0;
    uint32_t sender_groups_count =
        equal_batches_per_core ? (num_batches / num_batches_per_core_group_1) : num_virtual_rows;
    for (uint32_t i = 0; i < sender_groups_count; ++i) {
        uint32_t core_index = core_index_offset;
        for (uint32_t j = 0; j < num_groups / num_groups_per_core; ++j) {
            mcast_sender_core_ranges_all.insert(CoreRange(core_coords[core_index]));
            if (equal_batches_per_core || (virtual_core_coords[core_index].y <= last_row_with_extra_batch)) {
                mcast_sender_core_ranges_group_1.insert(CoreRange(core_coords[core_index]));
            } else {
                mcast_sender_core_ranges_group_2.insert(CoreRange(core_coords[core_index]));
            }
            core_index += num_virtual_rows;
        }
        core_index_offset += num_cores_per_batch;
    }
    CoreRangeSet mcast_sender_cores_group_1 = CoreRangeSet(mcast_sender_core_ranges_group_1);
    CoreRangeSet mcast_sender_cores_group_2 = CoreRangeSet(mcast_sender_core_ranges_group_2);

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
    auto reduce_sender_semaphore_id = tt::tt_metal::CreateSemaphore(program, all_cores, INVALID);

    std::map<std::string, std::string> reader_mcast_sender_defines;
    if (gamma.has_value()) {
        reader_mcast_sender_defines["FUSE_GAMMA"] = "1";
    }
    if (beta.has_value()) {
        reader_mcast_sender_defines["FUSE_BETA"] = "1";
    }
    if (reader_repack_output) {
        reader_mcast_sender_defines["READER_REPACK"] = "1";
    }
    if (tilize_in) {
        reader_mcast_sender_defines["TILIZE_IN"] = "1";
    }
    if (untilize_out) {
        reader_mcast_sender_defines["UNTILIZE_OUT"] = "1";
    }

    std::unordered_map<std::string, uint32_t> reader_mcast_sender_named_compile_time_args_group_1 = {
        {"reduce_receiver_semaphore_id", 0},
        {"reduce_sender_semaphore_id", reduce_sender_semaphore_id},
        {"num_cores_per_mcast_group", num_cores_per_mcast_group},
        {"num_batch_group", num_groups_per_core * num_batches_per_core_group_1},
        {"num_batches", num_batches_per_core_group_1},
        {"per_core_N", per_core_Nt},
        {"per_core_N_bytes", per_core_N_bytes_padded},
        {"per_core_N_bytes_with_stride", per_core_Nt * TILE_WIDTH * datum_size_bytes},
        {"datum_size_bytes", datum_size_bytes},
        {"per_core_M", per_core_Mt_group_1},
        {"TILE_HEIGHT", TILE_HEIGHT},
        {"block_h", block_ht_group_1},
        {"block_w", block_wt},
        {"block_hw", block_ht_group_1 * block_wt},
        {"num_cols_per_group", num_channels_per_group_mod_tile_w},
        {"num_tiles_per_batch", per_core_Mt_group_1 * Wt / num_batches_per_core_group_1},
        {"block_w_last", block_wt_last},
        {"GROUP_SIZE_IS_POWER_OF_2",
         (num_channels_per_group_mod_tile_w & (num_channels_per_group_mod_tile_w - 1)) == 0},
        {"GROUP_SIZE_SMALLER_THAN_TILE_W", num_channels_per_group < TILE_WIDTH},
        {"group_row_offset", num_channels_per_group - ((block_wt - 1) * TILE_WIDTH)},
        {"num_out_blocks", num_out_blocks},
        {"num_channels_per_group", num_channels_per_group},
        {"num_rows_per_group", num_rows_per_batch_per_core_group_1},
    };

    std::unordered_map<std::string, uint32_t> reader_mcast_sender_named_compile_time_args_group_2 = {
        {"reduce_receiver_semaphore_id", 0},
        {"reduce_sender_semaphore_id", reduce_sender_semaphore_id},
        {"num_cores_per_mcast_group", num_cores_per_mcast_group},
        {"num_batch_group", num_groups_per_core * num_batches_per_core_group_2},
        {"num_batches", num_batches_per_core_group_2},
        {"per_core_N", per_core_Nt},
        {"per_core_N_bytes", per_core_N_bytes_padded},
        {"per_core_N_bytes_with_stride", per_core_Nt * TILE_WIDTH * datum_size_bytes},
        {"datum_size_bytes", datum_size_bytes},
        {"per_core_M", per_core_Mt_group_2},
        {"TILE_HEIGHT", TILE_HEIGHT},
        {"block_h", block_ht_group_2},
        {"block_w", block_wt},
        {"block_hw", block_ht_group_2 * block_wt},
        {"num_cols_per_group", num_channels_per_group_mod_tile_w},
        {"num_tiles_per_batch", per_core_Mt_group_2 * Wt / num_batches_per_core_group_2},
        {"block_w_last", block_wt_last},
        {"GROUP_SIZE_IS_POWER_OF_2",
         (num_channels_per_group_mod_tile_w & (num_channels_per_group_mod_tile_w - 1)) == 0},
        {"GROUP_SIZE_SMALLER_THAN_TILE_W", num_channels_per_group < TILE_WIDTH},
        {"group_row_offset", num_channels_per_group - ((block_wt - 1) * TILE_WIDTH)},
        {"num_out_blocks", num_out_blocks},
        {"num_channels_per_group", num_channels_per_group},
        {"num_rows_per_group", num_rows_per_batch_per_core_group_2},
    };

    std::vector<uint32_t> reader_mcast_sender_compile_time_args_group_1 = {};
    std::vector<uint32_t> reader_mcast_sender_compile_time_args_group_2 = {};
    tt::tt_metal::TensorAccessorArgs(a.buffer()).append_to(reader_mcast_sender_compile_time_args_group_1);
    tt::tt_metal::TensorAccessorArgs(output.buffer()).append_to(reader_mcast_sender_compile_time_args_group_1);
    tt::tt_metal::TensorAccessorArgs(a.buffer()).append_to(reader_mcast_sender_compile_time_args_group_2);
    tt::tt_metal::TensorAccessorArgs(output.buffer()).append_to(reader_mcast_sender_compile_time_args_group_2);
    tt::tt_metal::NOC writer_noc = tt::tt_metal::detail::preferred_noc_for_dram_write(device->arch());
    tt::tt_metal::NOC reader_noc = tt::tt_metal::detail::preferred_noc_for_dram_read(device->arch());

    auto reader_mcast_sender_kernels_id_group_1 = CreateKernel(
        program,
        (use_welford ? "ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/dataflow/"
                       "welford_reader_mcast_sender_unary_gn.cpp"
                     : "ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/dataflow/"
                       "reader_mcast_sender_unary_gn.cpp"),
        mcast_sender_cores_group_1,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = reader_noc,
            .compile_args = reader_mcast_sender_compile_time_args_group_1,
            .defines = reader_mcast_sender_defines,
            .named_compile_args = reader_mcast_sender_named_compile_time_args_group_1,
        });
    auto reader_mcast_sender_kernels_id_group_2 = CreateKernel(
        program,
        (use_welford ? "ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/dataflow/"
                       "welford_reader_mcast_sender_unary_gn.cpp"
                     : "ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/dataflow/"
                       "reader_mcast_sender_unary_gn.cpp"),
        mcast_sender_cores_group_2,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = reader_noc,
            .compile_args = reader_mcast_sender_compile_time_args_group_2,
            .defines = reader_mcast_sender_defines,
            .named_compile_args = reader_mcast_sender_named_compile_time_args_group_2});

    std::map<std::string, std::string> writer_defines;
    std::vector<uint32_t> writer_mcast_sender_compile_time_args_group_1 = {};
    std::vector<uint32_t> writer_mcast_sender_compile_time_args_group_2 = {};
    tt::tt_metal::TensorAccessorArgs(output.buffer()).append_to(writer_mcast_sender_compile_time_args_group_1);
    tt::tt_metal::TensorAccessorArgs(gamma.has_value() ? gamma.value().buffer() : nullptr)
        .append_to(writer_mcast_sender_compile_time_args_group_1);
    tt::tt_metal::TensorAccessorArgs(beta.has_value() ? beta.value().buffer() : nullptr)
        .append_to(writer_mcast_sender_compile_time_args_group_1);
    tt::tt_metal::TensorAccessorArgs(input_mask.has_value() ? input_mask.value().buffer() : nullptr)
        .append_to(writer_mcast_sender_compile_time_args_group_1);

    tt::tt_metal::TensorAccessorArgs(output.buffer()).append_to(writer_mcast_sender_compile_time_args_group_2);
    tt::tt_metal::TensorAccessorArgs(gamma.has_value() ? gamma.value().buffer() : nullptr)
        .append_to(writer_mcast_sender_compile_time_args_group_2);
    tt::tt_metal::TensorAccessorArgs(beta.has_value() ? beta.value().buffer() : nullptr)
        .append_to(writer_mcast_sender_compile_time_args_group_2);
    tt::tt_metal::TensorAccessorArgs(input_mask.has_value() ? input_mask.value().buffer() : nullptr)
        .append_to(writer_mcast_sender_compile_time_args_group_2);

    std::unordered_map<std::string, uint32_t> writer_named_compile_time_args_group_1 = {
        {"is_mcast_sender", 1},
        {"fuse_gamma", gamma.has_value()},
        {"fuse_beta", beta.has_value()},
        {"num_cols_tile_gamma_beta", gamma_beta_num_cols_tile_per_core},
        {"per_core_M", per_core_Mt_group_1},
        {"per_core_N", per_core_Nt},
        {"per_core_N_bytes", per_core_N * datum_size_bytes},
        {"per_core_N_bytes_with_stride", per_core_Nt * TILE_WIDTH * datum_size_bytes},
        {"num_groups_per_core", num_groups_per_core},
        {"num_batches_per_core", num_batches_per_core_group_1},
        {"num_cols_per_group", num_channels_per_group_mod_tile_w},
        {"num_tiles_per_batch", per_core_Mt_group_1 * Wt / num_batches_per_core_group_1},
        {"block_w_last", block_wt_last},
        {"GROUP_SIZE_IS_POWER_OF_2",
         (num_channels_per_group_mod_tile_w & (num_channels_per_group_mod_tile_w - 1)) == 0},
        {"GROUP_SIZE_SMALLER_THAN_TILE_W", num_channels_per_group < TILE_WIDTH},
        {"group_row_offset", num_channels_per_group - ((block_wt - 1) * TILE_WIDTH)},
        {"num_out_blocks", num_out_blocks},
        {"block_h", block_ht_group_1},
        {"block_w", block_wt},
        {"block_hw", block_ht_group_1 * block_wt},
        {"groupnorm_mode", groupnorm_mode},
    };

    std::unordered_map<std::string, uint32_t> writer_named_compile_time_args_group_2 = {
        {"is_mcast_sender", 1},
        {"fuse_gamma", gamma.has_value()},
        {"fuse_beta", beta.has_value()},
        {"num_cols_tile_gamma_beta", gamma_beta_num_cols_tile_per_core},
        {"per_core_M", per_core_Mt_group_2},
        {"per_core_N", per_core_Nt},
        {"per_core_N_bytes", per_core_N * datum_size_bytes},
        {"per_core_N_bytes_with_stride", per_core_Nt * TILE_WIDTH * datum_size_bytes},
        {"num_groups_per_core", num_groups_per_core},
        {"num_batches_per_core", num_batches_per_core_group_2},
        {"num_cols_per_group", num_channels_per_group_mod_tile_w},
        {"num_tiles_per_batch", per_core_Mt_group_2 * Wt / num_batches_per_core_group_2},
        {"block_w_last", block_wt_last},
        {"GROUP_SIZE_IS_POWER_OF_2",
         (num_channels_per_group_mod_tile_w & (num_channels_per_group_mod_tile_w - 1)) == 0},
        {"GROUP_SIZE_SMALLER_THAN_TILE_W", num_channels_per_group < TILE_WIDTH},
        {"group_row_offset", num_channels_per_group - ((block_wt - 1) * TILE_WIDTH)},
        {"num_out_blocks", num_out_blocks},
        {"block_h", block_ht_group_2},
        {"block_w", block_wt},
        {"block_hw", block_ht_group_2 * block_wt},
        {"groupnorm_mode", groupnorm_mode},
    };

    if (gamma.has_value() && gamma.value().layout() == Layout::ROW_MAJOR) {
        auto gamma_stick_size = gamma.value().padded_shape()[3] * gamma.value().element_size();
        writer_named_compile_time_args_group_1["page_size"] = gamma_stick_size;
        writer_named_compile_time_args_group_2["page_size"] = gamma_stick_size;
    } else if (beta.has_value() && beta.value().layout() == Layout::ROW_MAJOR) {
        auto beta_stick_size = beta.value().padded_shape()[3] * beta.value().element_size();
        writer_named_compile_time_args_group_1["page_size"] = beta_stick_size;
        writer_named_compile_time_args_group_2["page_size"] = beta_stick_size;
    } else {
        writer_named_compile_time_args_group_1["page_size"] = TILE_HW * datum_size_bytes;
        writer_named_compile_time_args_group_2["page_size"] = TILE_HW * datum_size_bytes;
    }

    std::string writer_kernel =
        (use_welford ? "ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/dataflow/"
                       "welford_writer_unary_gn_rm_gb.cpp"
                     : "ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/dataflow/"
                       "writer_unary_gn_rm_gb.cpp");
    auto writer_kernels_id_group_1 = CreateKernel(
        program,
        writer_kernel,
        all_cores_group_1,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = writer_noc,
            .compile_args = writer_mcast_sender_compile_time_args_group_1,
            .defines = writer_defines,
            .named_compile_args = writer_named_compile_time_args_group_1});
    auto writer_kernels_id_group_2 = CreateKernel(
        program,
        writer_kernel,
        all_cores_group_2,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = writer_noc,
            .compile_args = writer_mcast_sender_compile_time_args_group_2,
            .defines = writer_defines,
            .named_compile_args = writer_named_compile_time_args_group_2});

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
    std::vector<uint32_t> mcast_sender_compute_compile_time_args_group_2 = {};

    std::unordered_map<std::string, uint32_t> mcast_sender_compute_named_compile_time_args_group_1 = {
        {"is_mcast_sender", 1},
        {"do_gamma", gamma.has_value()},
        {"do_beta", beta.has_value()},
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
        {"per_core_N_tile_bytes", per_core_Nt * TILE_HW * datum_size_bytes},
        {"num_groups_per_reset", num_groups_per_reset},
        {"single_tile_size_bytes", single_tile_size},
        {"num_tiles_per_batch", per_core_Mt_group_1 * Wt / num_batches_per_core_group_1},
        {"num_tiles_input_mask", num_groups_per_core * block_wt},
        {"num_cols_per_group", num_channels_per_group_mod_tile_w},
        {"block_w_last", block_wt_last},
        {"GROUP_SIZE_IS_POWER_OF_2",
         (num_channels_per_group_mod_tile_w & (num_channels_per_group_mod_tile_w - 1)) == 0},
        {"GROUP_SIZE_SMALLER_THAN_TILE_W", num_channels_per_group < TILE_WIDTH},
        {"group_row_offset", num_channels_per_group - ((block_wt - 1) * TILE_WIDTH)},
        {"num_out_blocks", num_out_blocks},
        {"num_channels_per_group", num_channels_per_group},
        {"num_rows_per_group", num_rows_per_batch_per_core_group_1},
        {"reciprocal_size", num_reciprocals},
    };

    std::unordered_map<std::string, uint32_t> mcast_sender_compute_named_compile_time_args_group_2 = {
        {"is_mcast_sender", 1},
        {"do_gamma", gamma.has_value()},
        {"do_beta", beta.has_value()},
        {"num_cores_per_mcast_group", num_cores_per_mcast_group},
        {"batch", num_batches_per_core_group_2},
        {"group", num_groups_per_core},
        {"block_h", block_ht_group_2},
        {"block_w", block_wt},
        {"block_hw", block_ht_group_2 * block_wt},
        {"subblock_w", subblock_wt},
        {"num_subblocks_w", num_subblocks_w},
        {"per_core_M", per_core_Mt_group_2},
        {"per_core_N", per_core_Nt},
        {"per_core_MN", per_core_Mt_group_2 * per_core_Nt},
        {"per_core_N_tile_bytes", per_core_Nt * TILE_HW * datum_size_bytes},
        {"num_groups_per_reset", num_groups_per_reset},
        {"single_tile_size_bytes", single_tile_size},
        {"num_tiles_per_batch", per_core_Mt_group_2 * Wt / num_batches_per_core_group_2},
        {"num_tiles_input_mask", num_groups_per_core * block_wt},
        {"num_cols_per_group", num_channels_per_group_mod_tile_w},
        {"block_w_last", block_wt_last},
        {"GROUP_SIZE_IS_POWER_OF_2",
         (num_channels_per_group_mod_tile_w & (num_channels_per_group_mod_tile_w - 1)) == 0},
        {"GROUP_SIZE_SMALLER_THAN_TILE_W", num_channels_per_group < TILE_WIDTH},
        {"group_row_offset", num_channels_per_group - ((block_wt - 1) * TILE_WIDTH)},
        {"num_out_blocks", num_out_blocks},
        {"num_channels_per_group", num_channels_per_group},
        {"num_rows_per_group", num_rows_per_batch_per_core_group_2},
        {"reciprocal_size", num_reciprocals},
    };

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);
    eltwise_binary_defines["FP32_DEST_ACC"] = fp32_dest_acc_en ? "true" : "false";
    CreateKernel(
        program,
        (use_welford ? "ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/compute/welford_groupnorm.cpp"
                     : "ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/compute/groupnorm.cpp"),
        mcast_sender_cores_group_1,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .math_approx_mode = math_approx_mode,
            .compile_args = mcast_sender_compute_compile_time_args_group_1,
            .defines = eltwise_binary_defines,
            .named_compile_args = mcast_sender_compute_named_compile_time_args_group_1,
        });
    CreateKernel(
        program,
        (use_welford ? "ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/compute/welford_groupnorm.cpp"
                     : "ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/compute/groupnorm.cpp"),
        mcast_sender_cores_group_2,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .math_approx_mode = math_approx_mode,
            .compile_args = mcast_sender_compute_compile_time_args_group_2,
            .defines = eltwise_binary_defines,
            .named_compile_args = mcast_sender_compute_named_compile_time_args_group_2,
        });

    // Create circular buffers
    uint32_t in0_cb_index = tt::CBIndex::c_0;
    uint32_t output_cb_index = tt::CBIndex::c_16;
    tt::tt_metal::CircularBufferConfig in0_cb_config_group_1 =
        tt::tt_metal::CircularBufferConfig(in0_CB_size_group_1, {{in0_cb_index, in_data_format}})
            .set_page_size(in0_cb_index, in_single_tile_size);
    tt::tt_metal::CircularBufferConfig output_cb_config_group_1 =
        tt::tt_metal::CircularBufferConfig(out_CB_size_group_1, {{output_cb_index, out_data_format}})
            .set_page_size(output_cb_index, out_single_tile_size);

    tt::tt_metal::CreateCircularBuffer(program, all_cores_group_1, in0_cb_config_group_1);
    tt::tt_metal::CreateCircularBuffer(program, all_cores_group_1, output_cb_config_group_1);
    tt::tt_metal::CircularBufferConfig in0_cb_config_group_2 =
        tt::tt_metal::CircularBufferConfig(in0_CB_size_group_2, {{in0_cb_index, in_data_format}})
            .set_page_size(in0_cb_index, in_single_tile_size);
    tt::tt_metal::CircularBufferConfig output_cb_config_group_2 =
        tt::tt_metal::CircularBufferConfig(out_CB_size_group_2, {{output_cb_index, out_data_format}})
            .set_page_size(output_cb_index, out_single_tile_size);

    tt::tt_metal::CreateCircularBuffer(program, all_cores_group_2, in0_cb_config_group_2);
    tt::tt_metal::CreateCircularBuffer(program, all_cores_group_2, output_cb_config_group_2);

    uint32_t in_cb_index = tt::CBIndex::c_29;
    tt::tt_metal::CircularBufferConfig in_cb_config_group_1 =
        tt::tt_metal::CircularBufferConfig(in_CB_size_group_1, {{in_cb_index, in_data_format}})
            .set_page_size(in_cb_index, in_single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores_group_1, in_cb_config_group_1);
    tt::tt_metal::CircularBufferConfig in_cb_config_group_2 =
        tt::tt_metal::CircularBufferConfig(in_CB_size_group_2, {{in_cb_index, in_data_format}})
            .set_page_size(in_cb_index, in_single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores_group_2, in_cb_config_group_2);

    if (untilize_out) {
        uint32_t out_cb_index = tt::CBIndex::c_30;
        tt::tt_metal::CircularBufferConfig out_cb_config_group_1 =
            tt::tt_metal::CircularBufferConfig(in_CB_size_group_1, {{out_cb_index, in_data_format}})
                .set_page_size(out_cb_index, in_single_tile_size);
        tt::tt_metal::CreateCircularBuffer(program, all_cores_group_1, out_cb_config_group_1);
        tt::tt_metal::CircularBufferConfig out_cb_config_group_2 =
            tt::tt_metal::CircularBufferConfig(in_CB_size_group_2, {{out_cb_index, in_data_format}})
                .set_page_size(out_cb_index, in_single_tile_size);
        tt::tt_metal::CreateCircularBuffer(program, all_cores_group_2, out_cb_config_group_2);
    }

    uint32_t in2_cb_index = tt::CBIndex::c_2;
    tt::tt_metal::CircularBufferConfig in2_cb_config =
        tt::tt_metal::CircularBufferConfig(in2_CB_size, {{in2_cb_index, cb_data_format}})
            .set_page_size(in2_cb_index, single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, in2_cb_config);

    uint32_t in3_cb_index = tt::CBIndex::c_3;
    tt::tt_metal::CircularBufferConfig in3_cb_config =
        tt::tt_metal::CircularBufferConfig(in3_CB_size, {{in3_cb_index, cb_data_format}})
            .set_page_size(in3_cb_index, single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, in3_cb_config);

    uint32_t in4_cb_index = tt::CBIndex::c_4;
    tt::tt_metal::CircularBufferConfig in4_cb_config =
        tt::tt_metal::CircularBufferConfig(in2_CB_size, {{in4_cb_index, cb_data_format}})
            .set_page_size(in4_cb_index, single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, in4_cb_config);

    if (gamma.has_value()) {
        uint32_t in5_cb_index = tt::CBIndex::c_5;
        tt::tt_metal::CircularBufferConfig in5_cb_config =
            tt::tt_metal::CircularBufferConfig(in5_CB_size, {{in5_cb_index, gamma_beta_cb_data_format}})
                .set_page_size(in5_cb_index, gamma_beta_single_tile_size);
        tt::tt_metal::CreateCircularBuffer(program, all_cores, in5_cb_config);
    }
    if (beta.has_value()) {
        uint32_t in6_cb_index = tt::CBIndex::c_6;
        tt::tt_metal::CircularBufferConfig in6_cb_config =
            tt::tt_metal::CircularBufferConfig(in6_CB_size, {{in6_cb_index, gamma_beta_cb_data_format}})
                .set_page_size(in6_cb_index, gamma_beta_single_tile_size);
        tt::tt_metal::CreateCircularBuffer(program, all_cores, in6_cb_config);
    }
    if (input_mask.has_value()) {
        uint32_t in_mask_cb_index = tt::CBIndex::c_28;
        tt::tt_metal::CircularBufferConfig in_mask_cb_config =
            tt::tt_metal::CircularBufferConfig(in_mask_CB_size, {{in_mask_cb_index, in_mask_cb_data_format}})
                .set_page_size(in_mask_cb_index, in_mask_single_tile_size);
        tt::tt_metal::CreateCircularBuffer(program, all_cores, in_mask_cb_config);
    }
    if (reader_repack_output) {
        uint32_t repack_cb_index = tt::CBIndex::c_26;
        uint32_t repack_out_cb_index = tt::CBIndex::c_31;
        std::map<uint8_t, tt::DataFormat> in0_out0_cb_data_format_spec{
            {repack_cb_index, in_data_format}, {repack_out_cb_index, in_data_format}};
        tt::tt_metal::CircularBufferConfig repack_cb_config =
            tt::tt_metal::CircularBufferConfig(repack_CB_size, in0_out0_cb_data_format_spec)
                .set_page_size(repack_cb_index, in_single_tile_size)
                .set_page_size(repack_out_cb_index, in_single_tile_size);
        tt::tt_metal::CreateCircularBuffer(program, all_cores, repack_cb_config);
    }

    uint32_t x_cb_index = tt::CBIndex::c_24;
    tt::tt_metal::CircularBufferConfig x_cb_config_group_1 =
        tt::tt_metal::CircularBufferConfig(x_CB_size_group_1, {{x_cb_index, cb_data_format}})
            .set_page_size(x_cb_index, single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores_group_1, x_cb_config_group_1);
    tt::tt_metal::CircularBufferConfig x_cb_config_group_2 =
        tt::tt_metal::CircularBufferConfig(x_CB_size_group_2, {{x_cb_index, cb_data_format}})
            .set_page_size(x_cb_index, single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores_group_2, x_cb_config_group_2);

    uint32_t xmm_cb_index = tt::CBIndex::c_25;
    tt::tt_metal::CircularBufferConfig xmm_cb_config_group_1 =
        tt::tt_metal::CircularBufferConfig(xmm_CB_size_group_1, {{xmm_cb_index, cb_data_format}})
            .set_page_size(xmm_cb_index, single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores_group_1, xmm_cb_config_group_1);
    tt::tt_metal::CircularBufferConfig xmm_cb_config_group_2 =
        tt::tt_metal::CircularBufferConfig(xmm_CB_size_group_2, {{xmm_cb_index, cb_data_format}})
            .set_page_size(xmm_cb_index, single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores_group_2, xmm_cb_config_group_2);

    uint32_t xmm2_cb_index = tt::CBIndex::c_23;
    tt::tt_metal::CircularBufferConfig xmm2_cb_config_group_1 =
        tt::tt_metal::CircularBufferConfig(xmm2_CB_size_group_1, {{xmm2_cb_index, cb_data_format}})
            .set_page_size(xmm2_cb_index, single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores_group_1, xmm2_cb_config_group_1);
    tt::tt_metal::CircularBufferConfig xmm2_cb_config_group_2 =
        tt::tt_metal::CircularBufferConfig(xmm2_CB_size_group_2, {{xmm2_cb_index, cb_data_format}})
            .set_page_size(xmm2_cb_index, single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores_group_2, xmm2_cb_config_group_2);

    uint32_t xmm3_cb_index = tt::CBIndex::c_22;
    tt::tt_metal::CircularBufferConfig xmm3_cb_config_group_1 =
        tt::tt_metal::CircularBufferConfig(xmm3_CB_size_group_1, {{xmm3_cb_index, cb_data_format}})
            .set_page_size(xmm3_cb_index, single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores_group_1, xmm3_cb_config_group_1);
    tt::tt_metal::CircularBufferConfig xmm3_cb_config_group_2 =
        tt::tt_metal::CircularBufferConfig(xmm3_CB_size_group_2, {{xmm3_cb_index, cb_data_format}})
            .set_page_size(xmm3_cb_index, single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores_group_2, xmm3_cb_config_group_2);

    uint32_t ex_cb_partial_index = tt::CBIndex::c_8;
    tt::tt_metal::CircularBufferConfig ex_cb_partial_config =
        tt::tt_metal::CircularBufferConfig(ex_partial_CB_size, {{ex_cb_partial_index, cb_data_format}})
            .set_page_size(ex_cb_partial_index, single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, ex_cb_partial_config);

    if (!use_welford) {
        uint32_t ex2_cb_partial_index = tt::CBIndex::c_21;
        tt::tt_metal::CircularBufferConfig ex2_cb_partial_config =
            tt::tt_metal::CircularBufferConfig(ex_partial_CB_size, {{ex2_cb_partial_index, cb_data_format}})
                .set_page_size(ex2_cb_partial_index, single_tile_size);
        tt::tt_metal::CreateCircularBuffer(program, all_cores, ex2_cb_partial_config);
    }

    uint32_t ex_cb_external_index = tt::CBIndex::c_10;
    tt::tt_metal::CircularBufferConfig ex_cb_external_config =
        tt::tt_metal::CircularBufferConfig(
            2 * single_tile_size * num_cores_per_mcast_group, {{ex_cb_external_index, cb_data_format}})
            .set_page_size(ex_cb_external_index, single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, ex_cb_external_config);

    uint32_t ex_cb_index = tt::CBIndex::c_9;
    uint32_t ex_global_cb_index = tt::CBIndex::c_15;
    std::map<uint8_t, tt::DataFormat> ex_global_cb_data_format_spec{
        {ex_global_cb_index, cb_data_format}, {ex_cb_index, cb_data_format}};
    auto ex_global_cb_config = tt::tt_metal::CircularBufferConfig(ex_global_CB_size, ex_global_cb_data_format_spec)
                                   .set_page_size(ex_global_cb_index, single_tile_size)
                                   .set_page_size(ex_cb_index, single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, ex_global_cb_config);

    if (!use_welford) {
        uint32_t ex2_cb_index = tt::CBIndex::c_13;
        uint32_t ex2_global_cb_index = tt::CBIndex::c_14;
        std::map<uint8_t, tt::DataFormat> ex2_global_cb_data_format_spec{
            {ex2_global_cb_index, cb_data_format}, {ex2_cb_index, cb_data_format}};
        auto ex2_global_cb_config =
            tt::tt_metal::CircularBufferConfig(ex2_global_CB_size, ex2_global_cb_data_format_spec)
                .set_page_size(ex2_global_cb_index, single_tile_size)
                .set_page_size(ex2_cb_index, single_tile_size);
        tt::tt_metal::CreateCircularBuffer(program, all_cores, ex2_global_cb_config);
    }

    uint32_t cb_ex2pe_index = tt::CBIndex::c_27;
    tt::tt_metal::CircularBufferConfig ex2pe_cb_config =
        tt::tt_metal::CircularBufferConfig(ex2pe_CB_size, {{cb_ex2pe_index, cb_data_format}})
            .set_page_size(cb_ex2pe_index, single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, ex2pe_cb_config);

    uint32_t cb_reciprocals = tt::CBIndex::c_18;
    CBHandle cb_reciprocals_handle = 0;
    if (reciprocals.has_value()) {
        tt::tt_metal::CircularBufferConfig reciprocal_cb_config =
            tt::tt_metal::CircularBufferConfig(reciprocal_CB_size, {{cb_reciprocals, reciprocal_cb_data_format}})
                .set_page_size(cb_reciprocals, reciprocal_CB_size)
                .set_globally_allocated_address(*reciprocals.value().buffer());
        cb_reciprocals_handle = tt::tt_metal::CreateCircularBuffer(program, all_cores, reciprocal_cb_config);
    }

    // Runtime Args
    std::vector<KernelHandle> writer_kernel_ids;
    std::vector<KernelHandle> reader_sender_kernel_ids;
    std::vector<KernelHandle> reader_receiver_kernel_ids;

    float winv_group_1 = 1.0f / std::sqrt(num_rows_per_batch_per_core_group_1 * num_channels_per_group);
    bfloat16 bfloat_winv_value_group_1 = bfloat16::truncate(winv_group_1);
    uint32_t packed_winv_value_group_1 =
        pack_two_bfloat16_into_uint32({bfloat_winv_value_group_1, bfloat_winv_value_group_1});
    float winv_group_2 = winv_group_1;
    bfloat16 bfloat_winv_value_group_2 = bfloat_winv_value_group_1;
    uint32_t packed_winv_value_group_2 = packed_winv_value_group_1;
    if (num_batches_per_core_group_2 > 0) {
        winv_group_2 = 1.0f / std::sqrt(num_rows_per_batch_per_core_group_2 * num_channels_per_group);
        bfloat_winv_value_group_2 = bfloat16::truncate(winv_group_2);
        packed_winv_value_group_2 =
            pack_two_bfloat16_into_uint32({bfloat_winv_value_group_2, bfloat_winv_value_group_2});
    }
    float cinv = 1.0f / std::sqrt(num_cores_per_batch * num_cores_per_group);
    bfloat16 bfloat_cinv_value = bfloat16::truncate(cinv);
    uint32_t packed_cinv_value = pack_two_bfloat16_into_uint32({bfloat_cinv_value, bfloat_cinv_value});
    union {
        float f;
        uint32_t u;
    } e{};
    e.f = eps;

    for (size_t i = 0; i < mcast_groups.size(); ++i) {
        auto group = mcast_groups[i];
        const auto& virtual_group = mcast_virtual_groups[i];
        bool rectangle_grid = is_rectangle_grid(group);

        for (size_t j = 0; j < group.size(); ++j) {
            CoreCoord core = group[j];
            CoreCoord virtual_core = virtual_group[j];
            uint32_t in0_start_id, out_tile_start_id;
            if (equal_batches_per_core || (virtual_core.y <= last_row_with_extra_batch)) {
                in0_start_id = per_core_Mt_group_1 * Wt * virtual_core.y + per_core_Nt * virtual_core.x;
                out_tile_start_id = per_core_Mt_group_1 * Wt * virtual_core.y + per_core_Nt * virtual_core.x;
            } else {
                in0_start_id = per_core_Mt_group_1 * Wt * (last_row_with_extra_batch + 1) +
                               per_core_Mt_group_2 * Wt * (virtual_core.y - last_row_with_extra_batch - 1) +
                               per_core_Nt * virtual_core.x;
                out_tile_start_id = in0_start_id;
            }

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
            mcast_sender_args.push_back((std::uint32_t)in0_dram_addr);
            mcast_sender_args.push_back((std::uint32_t)out_dram_addr);
            mcast_sender_args.push_back(in0_start_id);
            mcast_sender_args.push_back(out_tile_start_id);
            mcast_sender_args.push_back(Wt);
            mcast_sender_args.push_back(!mcast_group_first.empty());
            mcast_sender_args.push_back(!mcast_group_last.empty());
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
            for (const auto& core : group) {
                CoreCoord coord = device->worker_core_from_logical_core(core);
                mcast_noc_xy.push_back(coord.x);
            }
            for (const auto& core : group) {
                CoreCoord coord = device->worker_core_from_logical_core(core);
                mcast_noc_xy.push_back(coord.y);
            }
            mcast_sender_args.insert(mcast_sender_args.end(), mcast_noc_xy.begin(), mcast_noc_xy.end());
            if (equal_batches_per_core || (virtual_core.y <= last_row_with_extra_batch)) {
                tt::tt_metal::SetRuntimeArgs(program, reader_mcast_sender_kernels_id_group_1, core, mcast_sender_args);
                reader_sender_kernel_ids.push_back(reader_mcast_sender_kernels_id_group_1);
            } else {
                tt::tt_metal::SetRuntimeArgs(program, reader_mcast_sender_kernels_id_group_2, core, mcast_sender_args);
                reader_sender_kernel_ids.push_back(reader_mcast_sender_kernels_id_group_2);
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
        uint32_t out_tile_start_id;
        if (equal_batches_per_core || (virtual_core.y <= last_row_with_extra_batch)) {
            out_tile_start_id = per_core_Mt_group_1 * Wt * virtual_core.y + per_core_Nt * virtual_core.x;
        } else {
            out_tile_start_id = per_core_Mt_group_1 * Wt * (last_row_with_extra_batch + 1) +
                                per_core_Mt_group_2 * Wt * (virtual_core.y - last_row_with_extra_batch - 1) +
                                per_core_Nt * virtual_core.x;
        }
        if (virtual_core.x > curr_virtual_core_x) {
            curr_virtual_core_x++;
            if (gamma.has_value()) {
                gamma_tile_start_id = (gamma_tile_start_id + gamma_beta_num_cols_tile_per_core) %
                                      (gamma.value().physical_volume() / TILE_WIDTH);
            }
            if (beta.has_value()) {
                beta_tile_start_id = (beta_tile_start_id + gamma_beta_num_cols_tile_per_core) %
                                     (beta.value().physical_volume() / TILE_WIDTH);
            }
            if (input_mask.has_value()) {
                input_mask_tile_start_id = (input_mask_tile_start_id + input_mask_num_tiles_per_core) %
                                           (input_mask.value().physical_volume() / TILE_HW);
            }
        }

        std::vector<uint32_t> writer_mcast_sender_args;
        writer_mcast_sender_args.push_back(packed_cinv_value);
        if (equal_batches_per_core || (virtual_core.y <= last_row_with_extra_batch)) {
            writer_mcast_sender_args.push_back(packed_winv_value_group_1);
        } else {
            writer_mcast_sender_args.push_back(packed_winv_value_group_2);
        }
        writer_mcast_sender_args.push_back(e.u);
        writer_mcast_sender_args.push_back(out_dram_addr);
        writer_mcast_sender_args.push_back(gamma_dram_addr);
        writer_mcast_sender_args.push_back(beta_dram_addr);
        writer_mcast_sender_args.push_back(input_mask_dram_addr);
        writer_mcast_sender_args.push_back(out_tile_start_id);
        writer_mcast_sender_args.push_back(gamma_tile_start_id);
        writer_mcast_sender_args.push_back(beta_tile_start_id);
        writer_mcast_sender_args.push_back(input_mask_tile_start_id);
        writer_mcast_sender_args.push_back(Wt);
        if (equal_batches_per_core || (virtual_core.y <= last_row_with_extra_batch)) {
            tt::tt_metal::SetRuntimeArgs(program, writer_kernels_id_group_1, core, writer_mcast_sender_args);
            writer_kernel_ids.push_back(writer_kernels_id_group_1);
        } else {
            tt::tt_metal::SetRuntimeArgs(program, writer_kernels_id_group_2, core, writer_mcast_sender_args);
            writer_kernel_ids.push_back(writer_kernels_id_group_2);
        }
    }

    return cached_program_t{
        std::move(program),
        shared_variables_t{
            .writer_kernel_ids = writer_kernel_ids,
            .reader_sender_kernel_ids = reader_sender_kernel_ids,
            .reader_receiver_kernel_ids = reader_receiver_kernel_ids,
            .core_coords = core_coords,
            .grid_size = grid_size,
            .mcast_groups = mcast_groups,
            .groupnorm_mode = groupnorm_mode,
            .cb_reciprocals_handle = cb_reciprocals_handle}};
}

void GroupNormNoMcastProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const GroupNormParams& /*operation_attributes*/,
    const GroupNormInputs& tensor_args,
    Tensor& tensor_return_value) {
    auto& program = cached_program.program;
    auto& shared_vars = cached_program.shared_variables;

    auto src_buffer_a = tensor_args.input.buffer()->address();
    auto dst_buffer = tensor_return_value.buffer()->address();

    const auto& gamma = tensor_args.gamma;
    const auto& beta = tensor_args.beta;
    const auto& mask = tensor_args.input_mask;
    const auto& reciprocals = tensor_args.reciprocals;

    if (shared_vars.groupnorm_mode == 2 && reciprocals.has_value()) {
        UpdateDynamicCircularBufferAddress(program, shared_vars.cb_reciprocals_handle, *reciprocals.value().buffer());
    }

    for (uint32_t i = 0; i < shared_vars.core_coords.size(); ++i) {
        CoreCoord core = shared_vars.core_coords[i];
        auto writer_kernel_id = shared_vars.writer_kernel_ids.at(i);
        auto& writer_runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);

        writer_runtime_args[3] = dst_buffer;
        if (gamma.has_value()) {
            writer_runtime_args[4] = gamma.value().buffer()->address();
        }
        if (beta.has_value()) {
            writer_runtime_args[5] = beta.value().buffer()->address();
        }
        if (mask.has_value()) {
            writer_runtime_args[6] = mask.value().buffer()->address();
        }
    }

    uint32_t sender_index = 0;
    uint32_t receiver_index = 0;
    for (size_t i = 0; i < shared_vars.mcast_groups.size(); ++i) {
        const auto& group = shared_vars.mcast_groups[i];
        for (size_t j = 0; j < group.size(); ++j) {
            CoreCoord core = group[j];
            if (j == 0) {
                auto reader_sender_kernel_id = shared_vars.reader_sender_kernel_ids.at(sender_index);
                auto& reader_sender_runtime_args = GetRuntimeArgs(program, reader_sender_kernel_id, core);
                reader_sender_runtime_args[0] = src_buffer_a;
                reader_sender_runtime_args[1] = dst_buffer;
                sender_index++;
            } else {
                auto reader_receiver_kernel_id = shared_vars.reader_receiver_kernel_ids.at(receiver_index);
                auto& reader_receiver_runtime_args = GetRuntimeArgs(program, reader_receiver_kernel_id, core);
                reader_receiver_runtime_args[0] = src_buffer_a;
                reader_receiver_runtime_args[1] = dst_buffer;
                receiver_index++;
            }
        }
    }
}

}  // namespace ttnn::operations::normalization::group_norm

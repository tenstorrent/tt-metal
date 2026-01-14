// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "groupnorm_sharded_program_factory.hpp"
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

GroupNormShardedProgramFactory::cached_program_t GroupNormShardedProgramFactory::create(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args, Tensor& tensor_return_value) {
    const auto& a = tensor_args.input;
    const auto& gamma = tensor_args.gamma;
    const auto& beta = tensor_args.beta;
    const auto& input_mask = tensor_args.input_mask;
    const auto& negative_mask = tensor_args.negative_mask;
    auto& output = tensor_return_value;

    const auto& program_config = std::get<GroupNormShardedMultiCoreProgramConfig>(operation_attributes.program_config);
    float eps = operation_attributes.eps;
    uint32_t num_groups = operation_attributes.num_groups;
    uint32_t num_batches = a.padded_shape()[0];
    DataType im_data_format = program_config.im_data_format;
    CoreCoord grid_size = program_config.compute_with_storage_grid_size;
    bool inplace = program_config.inplace;
    bool use_welford = operation_attributes.use_welford;
    const auto& compute_kernel_config = operation_attributes.compute_kernel_config;

    // Begin sharded implementation
    if (gamma.has_value()) {
        TT_FATAL(
            gamma.value().layout() == Layout::ROW_MAJOR,
            "Gamma tensor must have ROW_MAJOR layout, but has {} layout",
            gamma.value().layout());
    }
    if (beta.has_value()) {
        TT_FATAL(
            beta.value().layout() == Layout::ROW_MAJOR,
            "Beta tensor must have ROW_MAJOR layout, but has {} layout",
            beta.value().layout());
    }

    bool is_height_sharding = a.padded_shape()[3] == a.shard_spec().value().shape[1];
    // convert data format
    tt::DataFormat in_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());
    tt::DataFormat out_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(im_data_format);
    tt::DataFormat gamma_beta_cb_data_format = tt::DataFormat::Float16_b;
    if (gamma.has_value()) {
        gamma_beta_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(gamma.value().dtype());
    }
    if (beta.has_value()) {
        gamma_beta_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(beta.value().dtype());
    }
    tt::DataFormat in_mask_cb_data_format =
        input_mask.has_value() ? tt::tt_metal::datatype_to_dataformat_converter(input_mask.value().dtype())
                               : tt::DataFormat::Float16_b;
    tt::DataFormat in_negative_mask_cb_data_format =
        negative_mask.has_value() ? tt::tt_metal::datatype_to_dataformat_converter(negative_mask.value().dtype())
                                  : tt::DataFormat::Float16_b;
    uint32_t datum_size_bytes = 2;  // bfloat16

    TT_FATAL(
        out_data_format == in_data_format,
        "Input and output must have the same data format, but input has {} and output has {}",
        in_data_format,
        out_data_format);

    // tile sizes
    uint32_t in_single_tile_size = tt::tile_size(in_data_format);
    uint32_t single_tile_size = tt::tile_size(cb_data_format);
    uint32_t out_single_tile_size = tt::tile_size(out_data_format);
    uint32_t gamma_beta_single_tile_size = tt::tile_size(gamma_beta_cb_data_format);
    uint32_t in_mask_single_tile_size = tt::tile_size(in_mask_cb_data_format);
    uint32_t in_negative_mask_single_tile_size = tt::tile_size(in_negative_mask_cb_data_format);
    // shard shape per core
    uint32_t per_core_M = a.shard_spec().value().shape[0];
    uint32_t per_core_N = a.shard_spec().value().shape[1];
    uint32_t per_core_Mt = per_core_M / TILE_HEIGHT;
    uint32_t per_core_Nt = (per_core_N + TILE_WIDTH - 1) / TILE_WIDTH;
    uint32_t per_core_N_bytes_padded = tt::round_up(per_core_N * datum_size_bytes, output.buffer()->alignment());
    bool reader_repack_output = (per_core_N % TILE_WIDTH) != 0;
    bool tilize_in = a.layout() == Layout::ROW_MAJOR;
    bool untilize_out = output.layout() == Layout::ROW_MAJOR;
    // tensor shape
    const auto& shape = a.padded_shape();
    uint32_t H = shape[2] * num_batches;
    uint32_t W = shape[3];
    uint32_t num_datum_row_per_group = W / num_groups;
    uint32_t num_datum_row_per_group_mod_tile_w =
        num_datum_row_per_group % TILE_WIDTH == 0 ? TILE_WIDTH : num_datum_row_per_group % TILE_WIDTH;
    uint32_t group_size = W / num_groups;
    // grid
    uint32_t num_cores_c = grid_size.x;
    uint32_t num_cores_r = grid_size.y;
    auto all_cores = a.shard_spec().value().grid;
    uint32_t num_cores = all_cores.num_cores();
    auto shard_orientation = a.shard_spec().value().orientation;
    // split each batch into multiple cores
    uint32_t num_shards_r = H / per_core_M;
    uint32_t num_cores_per_batch = num_batches > num_shards_r ? 1 : num_shards_r / num_batches;
    uint32_t num_shards_c = W / per_core_N;
    uint32_t num_cores_per_group = num_groups > num_shards_c ? 1 : num_shards_c / num_groups;
    // each core contains multiple batches
    uint32_t num_batches_per_core = num_batches > num_shards_r ? num_batches / num_shards_r : 1;
    uint32_t num_groups_per_core = num_groups > num_shards_c ? num_groups / num_shards_c : 1;

    TT_FATAL(
        per_core_N % num_datum_row_per_group == 0,
        "per_core_N ({}) must be divisible by num_datum_row_per_group ({})",
        per_core_N,
        num_datum_row_per_group);
    TT_FATAL(
        per_core_M % TILE_HEIGHT == 0,
        "per_core_M ({}) must be divisible by TILE_HEIGHT ({})",
        per_core_M,
        TILE_HEIGHT);
    if (per_core_N != W) {
        if (shard_orientation == ShardOrientation::COL_MAJOR) {
            TT_FATAL(
                per_core_N * num_cores_r == W,
                "per_core_N ({}) * num_cores_r ({}) must equal total width W ({})",
                per_core_N,
                num_cores_r,
                W);
            TT_FATAL(
                per_core_M * num_cores_c == H,
                "per_core_M ({}) * num_cores_c ({}) must equal total height H ({})",
                per_core_M,
                num_cores_c,
                H);
        } else {
            TT_FATAL(
                per_core_N * num_cores_c == W,
                "per_core_N ({}) * num_cores_c ({}) must equal total width W ({})",
                per_core_N,
                num_cores_c,
                W);
            TT_FATAL(
                per_core_M * num_cores_r == H,
                "per_core_M ({}) * num_cores_r ({}) must equal total height H ({})",
                per_core_M,
                num_cores_r,
                H);
        }
    }

    TT_FATAL(
        per_core_M % TILE_HEIGHT == 0,
        "per_core_M ({}) must be divisible by TILE_HEIGHT ({})",
        per_core_M,
        TILE_HEIGHT);

    TT_FATAL(W % num_groups == 0, "Tensor W ({}) must be divisible by num_groups ({})", W, num_groups);
    TT_FATAL(H % per_core_M == 0, "H dim ({}) must be divisible by per_core_M ({})", H, per_core_M);
    TT_FATAL(W % per_core_N == 0, "W dim ({}) must be divisible by per_core_N ({})", W, per_core_N);
    if (num_batches >= num_shards_r) {
        TT_FATAL(
            num_batches % num_shards_r == 0,
            "num_batches ({}) must be divisible by number of cores in a full column ({})",
            num_batches,
            num_shards_r);
    } else {
        TT_FATAL(
            num_shards_r % num_batches == 0,
            "number of cores in a full column ({}) must be divisible by num_batches ({})",
            num_shards_r,
            num_batches);
    }
    if (num_groups >= num_shards_c) {
        TT_FATAL(
            num_groups % num_shards_c == 0,
            "num_groups ({}) must be divisible by number of cores in a full row ({})",
            num_groups,
            num_shards_c);
    } else {
        TT_FATAL(
            num_shards_c % num_groups == 0,
            "number of cores in a full row ({}) must be divisible by num_groups ({})",
            num_shards_c,
            num_groups);
    }

    TT_FATAL(
        (!use_welford) || (num_groups_per_core <= 16),
        "num_groups_per_core ({}) must be <= 16 when use_welfords is true. Increase the width of shard spec to address "
        "this.",
        num_groups_per_core);

    // subblock
    uint32_t num_rows_per_batch_per_core = per_core_M / num_batches_per_core;
    auto [block_wt, num_groups_per_reset] = find_max_tile_span(per_core_N, group_size);
    uint32_t block_ht = per_core_Mt / num_batches_per_core;
    uint32_t subblock_wt = get_max_subblock(block_wt, 8);
    uint32_t num_subblocks_w = block_wt / subblock_wt;
    bool block_wt_last = (per_core_Nt + num_groups_per_core - 1) / num_groups_per_core;

    log_debug(tt::LogOp, "num_cores: {}", num_cores);
    log_debug(tt::LogOp, "num_rows_per_batch_per_core: {}", per_core_M / num_batches_per_core);
    log_debug(tt::LogOp, "per_core_M: {}", per_core_M);
    log_debug(tt::LogOp, "per_core_N: {}", per_core_N);
    log_debug(tt::LogOp, "W: {}", W);
    log_debug(tt::LogOp, "H: {}", H);
    log_debug(tt::LogOp, "num_datum_row_per_group: {}", num_datum_row_per_group);
    log_debug(tt::LogOp, "num_batches: {}", num_batches);
    log_debug(tt::LogOp, "num_groups: {}", num_groups);
    log_debug(tt::LogOp, "num_cores_r: {}", num_cores_r);
    log_debug(tt::LogOp, "num_cores_c: {}", num_cores_c);
    log_debug(tt::LogOp, "num_cores_per_batch: {}", num_cores_per_batch);
    log_debug(tt::LogOp, "num_cores_per_group: {}", num_cores_per_group);
    log_debug(tt::LogOp, "num_batches_per_core: {}", num_batches_per_core);
    log_debug(tt::LogOp, "num_groups_per_core: {}", num_groups_per_core);
    log_debug(tt::LogOp, "block_wt: {}", block_wt);
    log_debug(tt::LogOp, "block_wt_last: {}", block_wt_last);
    log_debug(tt::LogOp, "block_ht: {}", block_ht);
    log_debug(tt::LogOp, "subblock_wt: {}", subblock_wt);
    log_debug(tt::LogOp, "num_subblocks_w: {}", num_subblocks_w);
    log_debug(tt::LogOp, "reader_repack_output: {}", reader_repack_output);

    TT_FATAL(
        per_core_M % num_batches_per_core == 0,
        "shard height ({}) must be divisible by per_core_batch ({})",
        per_core_M,
        num_batches_per_core);
    TT_FATAL(W % num_groups == 0, "tensor width ({}) must be divisible by num_groups ({})", W, num_groups);
    if (shard_orientation == ShardOrientation::ROW_MAJOR && num_groups_per_core == 1) {
        TT_FATAL(
            num_cores_c % num_groups == 0,
            "for RM shard, when each group is split across cores, num_cores_c ({}) must be divisible by num_groups "
            "({})",
            num_cores_c,
            num_groups);
    } else if (shard_orientation == ShardOrientation::COL_MAJOR && num_groups_per_core == 1) {
        TT_FATAL(
            num_cores_r % num_groups == 0,
            "for CM shard, when each group is split across cores, num_cores_r ({}) must be divisible by num_groups "
            "({})",
            num_cores_r,
            num_groups);
    }

    if (per_core_N != W) {  // block sharded
        if (shard_orientation == ShardOrientation::ROW_MAJOR && num_batches_per_core == 1) {
            TT_FATAL(
                num_cores_r % num_batches == 0,
                "for RM shard, when each batch is split across cores, num_cores_r ({}) must be divisible by "
                "num_batches ({})",
                num_cores_r,
                num_batches);
        } else if (shard_orientation == ShardOrientation::COL_MAJOR && num_groups_per_core == 1) {
            TT_FATAL(
                num_cores_c % num_batches == 0,
                "for CM shard, when each batch is split across cores, num_cores_c ({}) must be divisible by "
                "num_batches ({})",
                num_cores_c,
                num_batches);
        }
    } else {  // height sharded
        if (num_batches_per_core == 1) {
            TT_FATAL(
                (num_cores_c * num_cores_r) % num_batches == 0,
                "for height shard, number of cores ({} * {} = {}) must be divisible by num_batches ({})",
                num_cores_c,
                num_cores_r,
                num_cores_c * num_cores_r,
                num_batches);
        }
    }

    if (input_mask.has_value()) {
        TT_FATAL(
            input_mask.value().padded_shape()[3] == block_wt * TILE_WIDTH,
            "input mask width ({}) must have the same width as block_wt * TILE_WIDTH ({})",
            input_mask.value().padded_shape()[3],
            block_wt * TILE_WIDTH);
    }

    // get sharded addr
    // gamma, beta addr
    auto gamma_dram_addr = gamma.has_value() ? gamma.value().buffer()->address() : 0;
    auto beta_dram_addr = beta.has_value() ? beta.value().buffer()->address() : 0;
    auto input_mask_dram_addr = input_mask.has_value() ? input_mask.value().buffer()->address() : 0;
    auto input_negative_mask_dram_addr = negative_mask.has_value() ? negative_mask.value().buffer()->address() : 0;

    ////////////////////////////////////////////////////////////////////////////
    //                      Grayskull Device Setup
    ////////////////////////////////////////////////////////////////////////////
    IDevice* device = a.device();

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    // block size for in0 (tensor a)
    uint32_t in0_block_tiles = per_core_Nt * per_core_Mt;
    uint32_t in0_CB_size = a.buffer()->aligned_size_per_bank();  // use buffer size to handle both RM and Tile
    uint32_t in_CB_size = in0_block_tiles * in_single_tile_size;
    // in2 - scaler
    uint32_t in2_CB_size = single_tile_size * (use_welford ? 2 : 1);
    // in3 - eps
    uint32_t in3_CB_size = single_tile_size;
    // gamma
    uint32_t gamma_beta_num_cols_tile_per_core = per_core_Nt;
    uint32_t in5_CB_size = gamma_beta_num_cols_tile_per_core * gamma_beta_single_tile_size;
    // beta
    uint32_t in6_CB_size = gamma_beta_num_cols_tile_per_core * gamma_beta_single_tile_size;
    // input mask
    uint32_t input_mask_num_tiles_per_core = block_wt * num_groups_per_core;
    uint32_t in_mask_CB_size =
        block_wt * in_mask_single_tile_size * (use_welford ? num_groups_per_core : 2);  // double buffer
    // negative mask
    uint32_t in_negative_mask_CB_size = block_wt * in_negative_mask_single_tile_size * 2;  // double buffer
    // repack cb
    uint32_t repack_CB_size = per_core_Nt * in_single_tile_size * 2;  // double buffer
    // itermediate buffers
    uint32_t interm_block_tiles = block_ht * block_wt;
    uint32_t x_CB_size = single_tile_size * (use_welford ? 1 : interm_block_tiles);
    // In welford, we both store mean and var here, so double the size
    uint32_t ex_partial_CB_size = single_tile_size * (use_welford ? 2 : 1);
    uint32_t ex_global_CB_size = ex_partial_CB_size * (use_welford ? num_groups_per_core : 1);  // the final result Ex
    uint32_t ex2pe_CB_size = use_welford ? single_tile_size * num_groups_per_core : ex_partial_CB_size;
    // output buffer size
    uint32_t out_CB_size = in0_block_tiles * out_single_tile_size;

    log_debug(tt::LogOp, "per_core_Nt: {}", per_core_Nt);
    log_debug(tt::LogOp, "per_core_Mt: {}", per_core_Mt);
    log_debug(tt::LogOp, "in0_CB_size: {}", in0_CB_size);
    log_debug(tt::LogOp, "in_CB_size: {}", in_CB_size);
    log_debug(tt::LogOp, "gamma_beta_num_cols_tile_per_core: {}", gamma_beta_num_cols_tile_per_core);
    log_debug(tt::LogOp, "in5_CB_size: {}", in5_CB_size);
    log_debug(tt::LogOp, "repack_CB_size: {}", repack_CB_size);

    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    Program program = Program();
    // define core ranges
    bool use_mcast = num_cores_per_batch > 1 || num_cores_per_group > 1;

    // create a vector of cores, in either RM or CM
    std::vector<CoreCoord> core_coords =
        grid_to_cores(num_cores, num_cores_c, num_cores_r, shard_orientation == ShardOrientation::ROW_MAJOR);
    for ([[maybe_unused]] const auto& core_coord : core_coords) {
        log_debug(tt::LogOp, "worker coord: {} {}", core_coord.x, core_coord.y);
    }
    std::vector<std::vector<CoreCoord>> core_coords2D;
    if (shard_orientation == ShardOrientation::ROW_MAJOR) {
        for (uint32_t i = 0; i < num_cores_c / num_cores_per_group; ++i) {
            for (uint32_t j = 0; j < num_cores_r; ++j) {
                std::vector<CoreCoord> temp;
                temp.reserve(num_cores_per_group);
                for (uint32_t k = 0; k < num_cores_per_group; ++k) {
                    temp.push_back(CoreCoord{(std::size_t)(k + (i * num_cores_per_group)), (std::size_t)j});
                }
                core_coords2D.push_back(temp);
            }
        }
    } else {
        for (uint32_t i = 0; i < num_cores_r / num_cores_per_group; ++i) {
            for (uint32_t j = 0; j < num_cores_c; ++j) {
                std::vector<CoreCoord> temp;
                temp.reserve(num_cores_per_group);
                for (uint32_t k = 0; k < num_cores_per_group; ++k) {
                    temp.push_back(CoreCoord{(std::size_t)j, (std::size_t)(k + (i * num_cores_per_group))});
                }
                core_coords2D.push_back(temp);
            }
        }
    }

    // one mcast core per batch per group
    std::set<CoreRange> mcast_sender_core_ranges;
    std::set<CoreRange> mcast_receiver_core_ranges;
    uint32_t core_index_offset = 0;
    for (uint32_t i = 0; i < num_batches / num_batches_per_core; ++i) {
        uint32_t core_index = core_index_offset;
        for (uint32_t j = 0; j < num_groups / num_groups_per_core; ++j) {
            mcast_sender_core_ranges.insert(CoreRange(core_coords[core_index]));
            core_index += num_cores_per_group;
            core_index_offset += num_cores_per_batch * num_cores_per_group;
        }
    }
    for ([[maybe_unused]] const auto& coord : mcast_sender_core_ranges) {
        log_debug(tt::LogOp, "mcast sender coord: {} {}", coord.start_coord.x, coord.start_coord.y);
    }
    for (uint32_t i = 0; i < num_cores; ++i) {
        // not found in mcast sender
        if (!mcast_sender_core_ranges.contains(CoreRange(core_coords[i]))) {
            mcast_receiver_core_ranges.insert(CoreRange(core_coords[i]));
        }
    }
    for ([[maybe_unused]] const auto& coord : mcast_receiver_core_ranges) {
        log_debug(tt::LogOp, "mcast receiver coord: {} {}", coord.start_coord.x, coord.start_coord.y);
    }
    CoreRangeSet mcast_sender_cores = CoreRangeSet(mcast_sender_core_ranges);
    CoreRangeSet mcast_receiver_cores = CoreRangeSet(mcast_receiver_core_ranges);
    // mcast groups
    std::vector<std::vector<CoreCoord>> mcast_groups;
    int group_index = -1;
    if (is_height_sharding) {
        for (uint32_t i = 0; i < num_cores; ++i) {
            if (mcast_sender_core_ranges.contains(CoreRange(core_coords[i]))) {
                group_index += 1;
            }
            if (group_index >= static_cast<int>(mcast_groups.size())) {
                mcast_groups.push_back(std::vector<CoreCoord>());  // Add a new group
            }
            mcast_groups[group_index].push_back(core_coords[i]);
        }
    } else {
        for (const auto& core_group : core_coords2D) {
            for (const auto& core : core_group) {
                if (mcast_sender_core_ranges.contains(CoreRange(core))) {
                    group_index += 1;
                }
                if (group_index >= static_cast<int>(mcast_groups.size())) {
                    mcast_groups.push_back(std::vector<CoreCoord>());  // Add a new group
                }
                mcast_groups[group_index].push_back(core);
            }
        }
    }
    for (size_t i = 0; i < mcast_groups.size(); ++i) {
        for (size_t j = 0; j < mcast_groups[i].size(); ++j) {
            log_debug(tt::LogOp, "mcast group: {} coord: {} {}", i, mcast_groups[i][j].x, mcast_groups[i][j].y);
        }
    }
    // how many cores in a mcast group
    uint32_t num_cores_per_mcast_group = mcast_groups[0].size();
    // Mcast args
    auto reduce_sender_semaphore_id = tt::tt_metal::CreateSemaphore(program, all_cores, INVALID);
    auto reduce_receiver_semaphore_id = tt::tt_metal::CreateSemaphore(program, all_cores, INVALID);
    // reader defines
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
    // reader compile time args
    std::vector<uint32_t> reader_mcast_sender_compile_time_args = {
        (std::uint32_t)reduce_receiver_semaphore_id,
        (std::uint32_t)reduce_sender_semaphore_id,
        (std::uint32_t)num_cores_per_mcast_group,
        (std::uint32_t)num_batches_per_core * (use_welford ? 1 : num_groups_per_core),
        (std::uint32_t)per_core_Nt,
        (std::uint32_t)per_core_N_bytes_padded,
        (std::uint32_t)per_core_Nt * TILE_WIDTH * datum_size_bytes,
        (std::uint32_t)datum_size_bytes,
        (std::uint32_t)per_core_Mt,
        (std::uint32_t)TILE_HEIGHT};
    if (use_welford) {
        reader_mcast_sender_compile_time_args.push_back(block_ht * block_wt);
        reader_mcast_sender_compile_time_args.push_back(num_groups_per_core);
    }
    std::vector<uint32_t> reader_mcast_receiver_compile_time_args = {
        (std::uint32_t)reduce_receiver_semaphore_id,
        (std::uint32_t)reduce_sender_semaphore_id,
        (std::uint32_t)num_batches_per_core * (use_welford ? 1 : num_groups_per_core),
        (std::uint32_t)per_core_Nt,
        (std::uint32_t)per_core_N_bytes_padded,
        (std::uint32_t)per_core_Nt * TILE_WIDTH * datum_size_bytes,
        (std::uint32_t)per_core_Mt,
        (std::uint32_t)TILE_HEIGHT};
    if (use_welford) {
        reader_mcast_receiver_compile_time_args.push_back(block_ht * block_wt);
        reader_mcast_receiver_compile_time_args.push_back(num_groups_per_core);
    }
    tt::tt_metal::NOC writer_noc = tt::tt_metal::detail::preferred_noc_for_dram_write(device->arch());
    tt::tt_metal::NOC reader_noc = tt::tt_metal::detail::preferred_noc_for_dram_read(device->arch());
    // reader kernel
    auto reader_mcast_sender_kernels_id = CreateKernel(
        program,
        (use_welford ? "ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/dataflow/"
                       "welford_reader_mcast_sender_unary_sharded_gn_v2.cpp"
                     : "ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/dataflow/"
                       "reader_mcast_sender_unary_sharded_gn_v2.cpp"),
        mcast_sender_cores,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = reader_noc,
            .compile_args = reader_mcast_sender_compile_time_args,
            .defines = reader_mcast_sender_defines});
    KernelHandle reader_mcast_receiver_kernels_id = -1;
    if (use_mcast) {
        reader_mcast_receiver_kernels_id = CreateKernel(
            program,
            (use_welford ? "ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/dataflow/"
                           "welford_reader_mcast_receiver_unary_sharded_gn_v2.cpp"
                         : "ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/dataflow/"
                           "reader_mcast_receiver_unary_sharded_gn_v2.cpp"),
            mcast_receiver_cores,
            tt::tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                .noc = reader_noc,
                .compile_args = reader_mcast_receiver_compile_time_args,
                .defines = reader_mcast_receiver_defines});
    }

    // writer defines
    std::map<std::string, std::string> writer_defines;
    if (negative_mask.has_value()) {
        writer_defines["FUSE_NEGATIVE_MASK"] = "1";
    }
    // writer compile time args
    std::vector<uint32_t> writer_mcast_sender_compile_time_args = {
        1,
        (std::uint32_t)gamma.has_value(),
        (std::uint32_t)beta.has_value(),
        (std::uint32_t)gamma_beta_num_cols_tile_per_core,
        (std::uint32_t)per_core_N,
        (std::uint32_t)per_core_N * datum_size_bytes,
        (std::uint32_t)per_core_Nt * TILE_WIDTH * datum_size_bytes,
        (std::uint32_t)num_groups_per_core,
        (std::uint32_t)num_batches_per_core,
        (std::uint32_t)block_wt};

    if (gamma.has_value() && gamma.value().layout() == Layout::ROW_MAJOR) {
        auto gamma_stick_size = gamma.value().padded_shape()[3] * gamma.value().element_size();
        writer_mcast_sender_compile_time_args.push_back(gamma_stick_size);
    } else if (beta.has_value() && beta.value().layout() == Layout::ROW_MAJOR) {
        auto beta_stick_size = beta.value().padded_shape()[3] * beta.value().element_size();
        writer_mcast_sender_compile_time_args.push_back(beta_stick_size);
    } else {
        writer_mcast_sender_compile_time_args.push_back(TILE_HW * datum_size_bytes);
    }

    // Append TensorAccessorArgs for sharded writer kernel
    tt::tt_metal::TensorAccessorArgs(gamma.has_value() ? gamma.value().buffer() : nullptr)
        .append_to(writer_mcast_sender_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(beta.has_value() ? beta.value().buffer() : nullptr)
        .append_to(writer_mcast_sender_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(input_mask.has_value() ? input_mask.value().buffer() : nullptr)
        .append_to(writer_mcast_sender_compile_time_args);

    // writer kernel
    if (negative_mask.has_value()) {
        TensorAccessorArgs(*negative_mask.value().buffer()).append_to(writer_mcast_sender_compile_time_args);
    } else {
        TensorAccessorArgs().append_to(writer_mcast_sender_compile_time_args);  // placeholder
    }

    // writer kernel
    std::string writer_kernel =
        (use_welford ? "ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/dataflow/"
                       "welford_writer_unary_sharded_gn_rm_gb_v2.cpp"
                     : "ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/dataflow/"
                       "writer_unary_sharded_gn_rm_gb_v2.cpp");
    auto writer_kernels_id = CreateKernel(
        program,
        writer_kernel,
        all_cores,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = writer_noc,
            .compile_args = writer_mcast_sender_compile_time_args,
            .defines = writer_defines});
    // defines
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
    if (negative_mask.has_value()) {
        eltwise_binary_defines["FUSE_NEGATIVE_MASK"] = "1";
    }
    // compute kernel compile time args
    std::vector<uint32_t> mcast_sender_compute_compile_time_args = {
        (std::uint32_t)1,
        (std::uint32_t)gamma.has_value(),
        (std::uint32_t)beta.has_value(),
        (std::uint32_t)num_cores_per_mcast_group,
        (std::uint32_t)num_batches_per_core,
        (std::uint32_t)num_groups_per_core,

        (std::uint32_t)num_datum_row_per_group_mod_tile_w,

        (std::uint32_t)block_ht,
        (std::uint32_t)block_wt,
        (std::uint32_t)block_ht * block_wt,

        (std::uint32_t)subblock_wt,
        (std::uint32_t)num_subblocks_w,

        (std::uint32_t)per_core_Mt,
        (std::uint32_t)per_core_Nt,
        (std::uint32_t)per_core_Mt * per_core_Nt,

        (std::uint32_t)per_core_Nt * TILE_HW * datum_size_bytes,  // per_core_N_tile_bytes
        (std::uint32_t)num_groups_per_reset,
        (std::uint32_t)single_tile_size,
        (std::uint32_t)per_core_Mt * per_core_Nt / num_batches_per_core,
        (std::uint32_t)num_groups_per_core * block_wt,
        (std::uint32_t)block_wt_last,
        (std::uint32_t)(num_datum_row_per_group_mod_tile_w & (num_datum_row_per_group_mod_tile_w - 1)) == 0,
        (std::uint32_t)num_datum_row_per_group < TILE_WIDTH,
        (std::uint32_t)num_datum_row_per_group - ((block_wt - 1) * TILE_WIDTH)};
    if (use_welford) {
        mcast_sender_compute_compile_time_args.push_back(num_datum_row_per_group);  // num_cols_per_group
    }
    std::vector<uint32_t> mcast_receiver_compute_compile_time_args = {
        (std::uint32_t)0,
        (std::uint32_t)gamma.has_value(),
        (std::uint32_t)beta.has_value(),
        (std::uint32_t)num_cores_per_mcast_group,
        (std::uint32_t)num_batches_per_core,
        (std::uint32_t)num_groups_per_core,

        (std::uint32_t)num_datum_row_per_group_mod_tile_w,

        (std::uint32_t)block_ht,
        (std::uint32_t)block_wt,
        (std::uint32_t)block_ht * block_wt,

        (std::uint32_t)subblock_wt,
        (std::uint32_t)num_subblocks_w,

        (std::uint32_t)per_core_Mt,
        (std::uint32_t)per_core_Nt,
        (std::uint32_t)per_core_Mt * per_core_Nt,

        (std::uint32_t)per_core_Nt * TILE_HW * datum_size_bytes,  // per_core_N_tile_bytes
        (std::uint32_t)num_groups_per_reset,
        (std::uint32_t)single_tile_size,
        (std::uint32_t)per_core_Mt * per_core_Nt / num_batches_per_core,
        (std::uint32_t)num_groups_per_core * block_wt,
        (std::uint32_t)block_wt_last,
        (std::uint32_t)(num_datum_row_per_group_mod_tile_w & (num_datum_row_per_group_mod_tile_w - 1)) == 0,
        (std::uint32_t)num_datum_row_per_group < TILE_WIDTH,
        (std::uint32_t)num_datum_row_per_group - ((block_wt - 1) * TILE_WIDTH)};
    if (use_welford) {
        mcast_receiver_compute_compile_time_args.push_back(num_datum_row_per_group);  // num_cols_per_group
    }
    // compute kernel
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);
    eltwise_binary_defines["FP32_DEST_ACC"] = fp32_dest_acc_en ? "true" : "false";
    CreateKernel(
        program,
        (use_welford
             ? "ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/compute/"
               "welford_groupnorm_sharded_v2.cpp"
             : "ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/compute/groupnorm_sharded_v2.cpp"),
        mcast_sender_cores,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .math_approx_mode = math_approx_mode,
            .compile_args = mcast_sender_compute_compile_time_args,
            .defines = eltwise_binary_defines});
    CreateKernel(
        program,
        (use_welford
             ? "ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/compute/"
               "welford_groupnorm_sharded_v2.cpp"
             : "ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/compute/groupnorm_sharded_v2.cpp"),
        mcast_receiver_cores,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .math_approx_mode = math_approx_mode,
            .compile_args = mcast_receiver_compute_compile_time_args,
            .defines = eltwise_binary_defines});
    // Create circular buffers
    uint32_t in0_cb_index = tt::CBIndex::c_0;
    uint32_t output_cb_index = tt::CBIndex::c_16;
    CBHandle cb_in0;
    CBHandle cb_output;
    if (inplace) {
        std::map<uint8_t, tt::DataFormat> in0_out0_cb_data_format_spec{
            {in0_cb_index, in_data_format}, {output_cb_index, in_data_format}};
        CircularBufferConfig in0_out0_cb_config =
            tt::tt_metal::CircularBufferConfig(in0_CB_size, in0_out0_cb_data_format_spec)
                .set_page_size(in0_cb_index, in_single_tile_size)
                .set_page_size(output_cb_index, in_single_tile_size)
                .set_globally_allocated_address(*a.buffer());

        cb_in0 = tt::tt_metal::CreateCircularBuffer(program, all_cores, in0_out0_cb_config);
        cb_output = cb_in0;
    } else {
        tt::tt_metal::CircularBufferConfig in0_cb_config =
            tt::tt_metal::CircularBufferConfig(in0_CB_size, {{in0_cb_index, in_data_format}})
                .set_page_size(in0_cb_index, in_single_tile_size)
                .set_globally_allocated_address(*a.buffer());

        tt::tt_metal::CircularBufferConfig output_cb_config =
            tt::tt_metal::CircularBufferConfig(out_CB_size, {{output_cb_index, out_data_format}})
                .set_page_size(output_cb_index, out_single_tile_size)
                .set_globally_allocated_address(*output.buffer());

        cb_in0 = tt::tt_metal::CreateCircularBuffer(program, all_cores, in0_cb_config);
        cb_output = tt::tt_metal::CreateCircularBuffer(program, all_cores, output_cb_config);
    }

    if (!negative_mask.has_value()) {
        // in - stores tilized input
        uint32_t in_cb_index = tt::CBIndex::c_1;
        tt::tt_metal::CircularBufferConfig in_cb_config =
            tt::tt_metal::CircularBufferConfig(in_CB_size, {{in_cb_index, in_data_format}})
                .set_page_size(in_cb_index, in_single_tile_size);
        tt::tt_metal::CreateCircularBuffer(program, all_cores, in_cb_config);
        if (untilize_out) {
            uint32_t out_cb_index = tt::CBIndex::c_30;
            tt::tt_metal::CircularBufferConfig out_cb_config =
                tt::tt_metal::CircularBufferConfig(in_CB_size, {{out_cb_index, in_data_format}})
                    .set_page_size(out_cb_index, in_single_tile_size);
            tt::tt_metal::CreateCircularBuffer(program, all_cores, out_cb_config);
        }
    } else {
        // in - stores tilized input
        // tilized in is overlapped with it
        uint32_t in_cb_index = tt::CBIndex::c_1;
        tt::tt_metal::CircularBufferConfig in_cb_config =
            tt::tt_metal::CircularBufferConfig(in_CB_size, {{in_cb_index, in_data_format}})
                .set_page_size(in_cb_index, in_single_tile_size);
        tt::tt_metal::CreateCircularBuffer(program, all_cores, in_cb_config);
    }
    // in2 scaler - for partial Ex
    uint32_t in2_cb_index = tt::CBIndex::c_2;
    tt::tt_metal::CircularBufferConfig in2_cb_config =
        tt::tt_metal::CircularBufferConfig(in2_CB_size, {{in2_cb_index, cb_data_format}})
            .set_page_size(in2_cb_index, single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, in2_cb_config);
    // in3 eps
    uint32_t in3_cb_index = tt::CBIndex::c_3;
    tt::tt_metal::CircularBufferConfig in3_cb_config =
        tt::tt_metal::CircularBufferConfig(in3_CB_size, {{in3_cb_index, cb_data_format}})
            .set_page_size(in3_cb_index, single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, in3_cb_config);
    // in4 scaler-c
    if (!use_welford) {
        uint32_t in4_cb_index = tt::CBIndex::c_4;
        tt::tt_metal::CircularBufferConfig in4_cb_config =
            tt::tt_metal::CircularBufferConfig(in2_CB_size, {{in4_cb_index, cb_data_format}})
                .set_page_size(in4_cb_index, single_tile_size);
        tt::tt_metal::CreateCircularBuffer(program, all_cores, in4_cb_config);
    }
    // gamma
    if (gamma.has_value()) {
        uint32_t in5_cb_index = tt::CBIndex::c_5;
        tt::tt_metal::CircularBufferConfig in5_cb_config =
            tt::tt_metal::CircularBufferConfig(in5_CB_size, {{in5_cb_index, gamma_beta_cb_data_format}})
                .set_page_size(in5_cb_index, gamma_beta_single_tile_size);
        tt::tt_metal::CreateCircularBuffer(program, all_cores, in5_cb_config);
    }
    // beta
    if (beta.has_value()) {
        uint32_t in6_cb_index = tt::CBIndex::c_6;
        tt::tt_metal::CircularBufferConfig in6_cb_config =
            tt::tt_metal::CircularBufferConfig(in6_CB_size, {{in6_cb_index, gamma_beta_cb_data_format}})
                .set_page_size(in6_cb_index, gamma_beta_single_tile_size);
        tt::tt_metal::CreateCircularBuffer(program, all_cores, in6_cb_config);
    }
    // input mask
    if (input_mask.has_value()) {
        uint32_t in_mask_cb_index = tt::CBIndex::c_7;
        tt::tt_metal::CircularBufferConfig in_mask_cb_config =
            tt::tt_metal::CircularBufferConfig(in_mask_CB_size, {{in_mask_cb_index, in_mask_cb_data_format}})
                .set_page_size(in_mask_cb_index, in_mask_single_tile_size);
        tt::tt_metal::CreateCircularBuffer(program, all_cores, in_mask_cb_config);
    }
    // negative mask
    if (negative_mask.has_value()) {
        uint32_t in_negative_mask_cb_index = tt::CBIndex::c_14;
        tt::tt_metal::CircularBufferConfig in_negative_mask_cb_config =
            tt::tt_metal::CircularBufferConfig(
                in_negative_mask_CB_size, {{in_negative_mask_cb_index, in_negative_mask_cb_data_format}})
                .set_page_size(in_negative_mask_cb_index, in_negative_mask_single_tile_size);
        tt::tt_metal::CreateCircularBuffer(program, all_cores, in_negative_mask_cb_config);
    }
    if (reader_repack_output) {
        uint32_t repack_cb_index = tt::CBIndex::c_11;
        uint32_t repack_out_cb_index = tt::CBIndex::c_12;
        std::map<uint8_t, tt::DataFormat> in0_out0_cb_data_format_spec{
            {repack_cb_index, in_data_format}, {repack_out_cb_index, in_data_format}};
        tt::tt_metal::CircularBufferConfig repack_cb_config =
            tt::tt_metal::CircularBufferConfig(repack_CB_size, in0_out0_cb_data_format_spec)
                .set_page_size(repack_cb_index, in_single_tile_size)
                .set_page_size(repack_out_cb_index, in_single_tile_size);
        tt::tt_metal::CreateCircularBuffer(program, all_cores, repack_cb_config);
    }
    // x
    uint32_t x_cb_index = tt::CBIndex::c_13;
    tt::tt_metal::CircularBufferConfig x_cb_config =
        tt::tt_metal::CircularBufferConfig(x_CB_size, {{x_cb_index, cb_data_format}})
            .set_page_size(x_cb_index, single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, x_cb_config);

    // ex_partial
    uint32_t ex_cb_partial_index = tt::CBIndex::c_8;
    tt::tt_metal::CircularBufferConfig ex_cb_partial_config =
        tt::tt_metal::CircularBufferConfig(ex_partial_CB_size, {{ex_cb_partial_index, cb_data_format}})
            .set_page_size(ex_cb_partial_index, single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, ex_cb_partial_config);

    // ex_external
    uint32_t ex_cb_external_index = tt::CBIndex::c_10;
    tt::tt_metal::CircularBufferConfig ex_cb_external_config =
        tt::tt_metal::CircularBufferConfig(single_tile_size, {{ex_cb_external_index, cb_data_format}})
            .set_page_size(ex_cb_external_index, single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, ex_cb_external_config);

    // ex_global
    uint32_t ex_cb_index = tt::CBIndex::c_9;
    uint32_t ex_global_cb_index = tt::CBIndex::c_15;
    std::map<uint8_t, tt::DataFormat> ex_global_cb_data_format_spec{
        {ex_global_cb_index, cb_data_format}, {ex_cb_index, cb_data_format}};
    auto ex_global_cb_config = tt::tt_metal::CircularBufferConfig(ex_global_CB_size, ex_global_cb_data_format_spec)
                                   .set_page_size(ex_global_cb_index, single_tile_size)
                                   .set_page_size(ex_cb_index, single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, ex_global_cb_config);

    // ex2pe
    uint32_t cb_ex2pe_index;
    cb_ex2pe_index = tt::CBIndex::c_17;
    tt::tt_metal::CircularBufferConfig ex2pe_cb_config =
        tt::tt_metal::CircularBufferConfig(ex2pe_CB_size, {{cb_ex2pe_index, cb_data_format}})
            .set_page_size(cb_ex2pe_index, single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, ex2pe_cb_config);

    uint32_t cb_ones_index = tt::CBIndex::c_26;
    tt::tt_metal::CircularBufferConfig ones_cb_config =
        tt::tt_metal::CircularBufferConfig(single_tile_size, {{cb_ones_index, cb_data_format}})
            .set_page_size(cb_ones_index, single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, ones_cb_config);

    // Runtime Args
    std::vector<KernelHandle> writer_kernel_ids;
    float winv = 1.0f / std::sqrt(num_rows_per_batch_per_core * num_datum_row_per_group);  // bcast-w scaler
    // TODO: #27672: Truncation should be removed once we figure a root cause of regression without it
    bfloat16 bfloat_winv_value = bfloat16::truncate(winv);
    uint32_t packed_winv_value = pack_two_bfloat16_into_uint32({bfloat_winv_value, bfloat_winv_value});
    float cinv = 1.0f / std::sqrt(num_cores_per_batch * num_cores_per_group);  // bcast-cores scaler
    bfloat16 bfloat_cinv_value = bfloat16::truncate(cinv);
    uint32_t packed_cinv_value = pack_two_bfloat16_into_uint32({bfloat_cinv_value, bfloat_cinv_value});
    union {
        float f;
        uint32_t u;
    } e{};
    e.f = eps;

    log_debug(tt::LogOp, "num_rows_per_batch_per_core: {}", num_rows_per_batch_per_core);
    log_debug(tt::LogOp, "num_datum_row_per_group: {}", num_datum_row_per_group);
    log_debug(tt::LogOp, "num_cores_per_batch: {}", num_cores_per_batch);
    log_debug(tt::LogOp, "num_cores_per_group: {}", num_cores_per_group);

    for (auto group : mcast_groups) {
        bool rectangle_grid = is_rectangle_grid(group);

        for (size_t j = 0; j < group.size(); ++j) {
            CoreCoord core = group[j];

            if (j == 0) {  // mcast sender
                // get the bounding box for the mcast
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
                mcast_sender_args.push_back(!mcast_group_first.empty());
                mcast_sender_args.push_back(!mcast_group_last.empty());
                mcast_sender_args.push_back(mcast_start.x);
                mcast_sender_args.push_back(mcast_start.y);
                mcast_sender_args.push_back(mcast_end.x);
                mcast_sender_args.push_back(mcast_end.y);
                if (!mcast_group_first.empty()) {
                    mcast_sender_args.push_back(mcast_group_mid.size());
                    log_debug(tt::LogOp, "mcast mid group size: {}", mcast_group_mid.size());
                } else {
                    mcast_sender_args.push_back(mcast_group_mid.size() - 1);  // mcast w/o itself
                    log_debug(tt::LogOp, "mcast mid group size: {}", mcast_group_mid.size() - 1);
                }

                log_debug(
                    tt::LogOp,
                    "mcast mid group start coord: {} {} end coord: {} {}",
                    mcast_start.x,
                    mcast_start.y,
                    mcast_end.x,
                    mcast_end.y);

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
                    mcast_sender_args.push_back(mcast_group_first.size() - 1);  // mcast w/0 itself

                    log_debug(
                        tt::LogOp,
                        "mcast first group start coord: {} {} end coord: {} {}",
                        mcast_first_start.x,
                        mcast_first_start.y,
                        mcast_first_end.x,
                        mcast_first_end.y);
                    log_debug(tt::LogOp, "mcast first group size: {}", mcast_group_first.size() - 1);
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

                    log_debug(
                        tt::LogOp,
                        "mcast last group start coord: {} {} end coord: {} {}",
                        mcast_last_start.x,
                        mcast_last_start.y,
                        mcast_last_end.x,
                        mcast_last_end.y);
                    log_debug(tt::LogOp, "mcast last group size: {}", mcast_group_last.size());
                }

                // add all coords within a group
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
                tt::tt_metal::SetRuntimeArgs(program, reader_mcast_sender_kernels_id, core, mcast_sender_args);

            } else {  // mcast receiver
                log_debug(tt::LogOp, "mcast receiver receive from coord: {} {}", group.front().x, group.front().y);
                std::vector<uint32_t> mcast_receiver_args;
                mcast_receiver_args.push_back(device->worker_core_from_logical_core(group.front()).x);
                mcast_receiver_args.push_back(device->worker_core_from_logical_core(group.front()).y);
                tt::tt_metal::SetRuntimeArgs(program, reader_mcast_receiver_kernels_id, core, mcast_receiver_args);
            }
        }
    }

    // writer
    uint32_t gamma_tile_start_id = 0;
    uint32_t beta_tile_start_id = 0;
    uint32_t input_mask_tile_start_id = 0;
    for (auto core : core_coords) {
        std::vector<uint32_t> writer_mcast_sender_args;
        writer_mcast_sender_args.push_back(packed_cinv_value);
        writer_mcast_sender_args.push_back(packed_winv_value);
        writer_mcast_sender_args.push_back(e.u);
        writer_mcast_sender_args.push_back(gamma_dram_addr);
        writer_mcast_sender_args.push_back(beta_dram_addr);
        writer_mcast_sender_args.push_back(input_mask_dram_addr);
        writer_mcast_sender_args.push_back(input_negative_mask_dram_addr);
        writer_mcast_sender_args.push_back(gamma_tile_start_id);
        writer_mcast_sender_args.push_back(beta_tile_start_id);
        writer_mcast_sender_args.push_back(input_mask_tile_start_id);
        tt::tt_metal::SetRuntimeArgs(program, writer_kernels_id, core, writer_mcast_sender_args);
        writer_kernel_ids.push_back(writer_kernels_id);

        if (gamma.has_value()) {
            gamma_tile_start_id = (gamma_tile_start_id + gamma_beta_num_cols_tile_per_core) %
                                  (gamma.value().physical_volume() / TILE_WIDTH);
        }
        if (beta.has_value()) {
            beta_tile_start_id = (beta_tile_start_id + gamma_beta_num_cols_tile_per_core) %
                                 (beta.value().physical_volume() / TILE_WIDTH);
        }
        if (input_mask.has_value()) {
            // Tile id for negative mask is same as input mask
            input_mask_tile_start_id = (input_mask_tile_start_id + input_mask_num_tiles_per_core) %
                                       (input_mask.value().physical_volume() / TILE_HW);
        }
    }

    return cached_program_t{
        std::move(program),
        shared_variables_t{
            .writer_kernel_ids = writer_kernel_ids,
            .cb_in0 = cb_in0,
            .cb_output = cb_output,
            .num_cores = num_cores,
            .grid_size = grid_size}};
}

void GroupNormShardedProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    Tensor& tensor_return_value) {
    auto& program = cached_program.program;
    auto& shared_vars = cached_program.shared_variables;
    const auto& input = tensor_args.input;
    auto& output = tensor_return_value;

    auto* src_buffer = input.buffer();
    auto* dst_buffer = output.buffer();

    UpdateDynamicCircularBufferAddress(program, shared_vars.cb_in0, *src_buffer);
    if (shared_vars.cb_output != shared_vars.cb_in0) {
        UpdateDynamicCircularBufferAddress(program, shared_vars.cb_output, *dst_buffer);
    }

    const auto& gamma = tensor_args.gamma;
    const auto& beta = tensor_args.beta;
    const auto& mask = tensor_args.input_mask;
    const auto& negative_mask = tensor_args.negative_mask;

    for (uint32_t i = 0; i < shared_vars.num_cores && i < shared_vars.writer_kernel_ids.size(); ++i) {
        CoreCoord core = {i % shared_vars.grid_size.x, i / shared_vars.grid_size.x};
        auto writer_kernel_id = shared_vars.writer_kernel_ids.at(i);
        auto& runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);

        if (gamma.has_value()) {
            runtime_args[3] = gamma.value().buffer()->address();
        }
        if (beta.has_value()) {
            runtime_args[4] = beta.value().buffer()->address();
        }
        if (mask.has_value()) {
            runtime_args[5] = mask.value().buffer()->address();
        }
        if (negative_mask.has_value()) {
            runtime_args[6] = negative_mask.value().buffer()->address();
        }
    }
}

}  // namespace ttnn::operations::normalization::group_norm

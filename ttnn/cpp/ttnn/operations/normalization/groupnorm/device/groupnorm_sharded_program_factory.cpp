// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "groupnorm_device_operation.hpp"
#include "groupnorm_program_utils.hpp"

#include <bit>
#include <map>
#include <string>
#include <optional>

#include <tt-metalium/circular_buffer_constants.h>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/operations/math.hpp"

using uint32_t = std::uint32_t;
using namespace tt::tt_metal;

namespace ttnn::prim {

tt::tt_metal::ProgramDescriptor GroupNormDeviceOperation::GroupNormShardedProgramFactory::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    const auto& a = tensor_args.input;
    const auto& gamma = tensor_args.gamma;
    const auto& beta = tensor_args.beta;
    const auto& input_mask = tensor_args.input_mask;
    const auto& negative_mask = tensor_args.negative_mask;
    auto& output = tensor_return_value;

    const uint32_t tile_height = a.tensor_spec().tile().get_height();
    const uint32_t tile_width = a.tensor_spec().tile().get_width();
    const uint32_t tile_hw = a.tensor_spec().tile().get_tile_hw();

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
    uint32_t per_core_Mt = per_core_M / tile_height;
    uint32_t per_core_Nt = (per_core_N + tile_width - 1) / tile_width;
    uint32_t per_core_N_bytes_padded = tt::round_up(per_core_N * datum_size_bytes, output.buffer()->alignment());
    bool reader_repack_output = (per_core_N % tile_width) != 0;
    bool tilize_in = a.layout() == Layout::ROW_MAJOR;
    bool untilize_out = output.layout() == Layout::ROW_MAJOR;
    // tensor shape
    const auto& shape = a.padded_shape();
    uint32_t H = shape[2] * num_batches;
    uint32_t W = shape[3];
    uint32_t num_datum_row_per_group = W / num_groups;
    uint32_t num_datum_row_per_group_mod_tile_w =
        num_datum_row_per_group % tile_width == 0 ? tile_width : num_datum_row_per_group % tile_width;
    uint32_t group_size = W / num_groups;
    auto all_cores = a.shard_spec().value().grid.merge_ranges();
    TT_FATAL(all_cores.ranges().size() == 1, "sharded groupnorm requires a rectangular shard grid");
    uint32_t num_cores = all_cores.num_cores();
    auto shard_orientation = a.shard_spec().value().orientation;
    const auto shard_bbox = all_cores.bounding_box();
    // grid
    uint32_t num_cores_c = shard_bbox.end_coord.x - shard_bbox.start_coord.x + 1;
    uint32_t num_cores_r = shard_bbox.end_coord.y - shard_bbox.start_coord.y + 1;
    TT_FATAL(
        grid_size.x == num_cores_c && grid_size.y == num_cores_r,
        "program_config compute_with_storage_grid_size ({}x{}) must match shard grid dimensions ({}x{})",
        grid_size.x,
        grid_size.y,
        num_cores_c,
        num_cores_r);
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
        per_core_M % tile_height == 0,
        "per_core_M ({}) must be divisible by tile_height ({})",
        per_core_M,
        tile_height);
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
        per_core_M % tile_height == 0,
        "per_core_M ({}) must be divisible by tile_height ({})",
        per_core_M,
        tile_height);

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
            input_mask.value().padded_shape()[3] == block_wt * tile_width,
            "input mask width ({}) must have the same width as block_wt * tile_width ({})",
            input_mask.value().padded_shape()[3],
            block_wt * tile_width);
    }

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
    uint32_t in2_CB_size = single_tile_size * (use_welford ? 3 : 1);
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
    // define core ranges
    bool use_mcast = num_cores_per_batch > 1 || num_cores_per_group > 1;

    // create a vector of cores, in either RM or CM
    std::vector<CoreCoord> core_coords =
        corerange_to_cores(all_cores, num_cores, shard_orientation == ShardOrientation::ROW_MAJOR);
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
                    const uint32_t idx = j * num_cores_c + i * num_cores_per_group + k;
                    temp.push_back(core_coords[idx]);
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
                    const uint32_t idx = j * num_cores_r + k + i * num_cores_per_group;
                    temp.push_back(core_coords[idx]);
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

    // ---- Build ProgramDescriptor ----
    ProgramDescriptor desc;

    // Mcast args - semaphore IDs assigned sequentially in descriptor.semaphores
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
        reduce_receiver_semaphore_id,
        reduce_sender_semaphore_id,
        num_cores_per_mcast_group,
        num_batches_per_core * (use_welford ? 1 : num_groups_per_core),
        per_core_Nt,
        per_core_N_bytes_padded,
        per_core_Nt * tile_width * datum_size_bytes,
        datum_size_bytes,
        per_core_Mt,
        tile_height};
    if (use_welford) {
        reader_mcast_sender_compile_time_args.push_back(block_ht * block_wt);
        reader_mcast_sender_compile_time_args.push_back(num_groups_per_core);
        reader_mcast_sender_compile_time_args.push_back(tile_width);
    }
    std::vector<uint32_t> reader_mcast_receiver_compile_time_args = {
        reduce_receiver_semaphore_id,
        reduce_sender_semaphore_id,
        num_batches_per_core * (use_welford ? 1 : num_groups_per_core),
        per_core_Nt,
        per_core_N_bytes_padded,
        per_core_Nt * tile_width * datum_size_bytes,
        per_core_Mt,
        tile_height};
    if (use_welford) {
        reader_mcast_receiver_compile_time_args.push_back(block_ht * block_wt);
        reader_mcast_receiver_compile_time_args.push_back(num_groups_per_core);
        reader_mcast_receiver_compile_time_args.push_back(tile_width);
    }
    tt::tt_metal::NOC writer_noc = tt::tt_metal::detail::preferred_noc_for_dram_write(device->arch());
    tt::tt_metal::NOC reader_noc = tt::tt_metal::detail::preferred_noc_for_dram_read(device->arch());

    // reader sender kernel
    KernelDescriptor reader_mcast_sender_desc;
    reader_mcast_sender_desc.kernel_source =
        (use_welford ? "ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/dataflow/"
                       "welford_reader_mcast_sender_unary_sharded_gn_v2.cpp"
                     : "ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/dataflow/"
                       "reader_mcast_sender_unary_sharded_gn_v2.cpp");
    reader_mcast_sender_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_mcast_sender_desc.core_ranges = mcast_sender_cores;
    reader_mcast_sender_desc.compile_time_args = reader_mcast_sender_compile_time_args;
    reader_mcast_sender_desc.defines =
        KernelDescriptor::Defines(reader_mcast_sender_defines.begin(), reader_mcast_sender_defines.end());
    reader_mcast_sender_desc.config = DataMovementConfigDescriptor{
        .processor = DataMovementProcessor::RISCV_0,
        .noc = reader_noc,
    };

    // reader receiver kernel (only when mcast is in use)
    KernelDescriptor reader_mcast_receiver_desc;
    bool has_receiver_kernel = use_mcast;
    if (has_receiver_kernel) {
        reader_mcast_receiver_desc.kernel_source =
            (use_welford ? "ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/dataflow/"
                           "welford_reader_mcast_receiver_unary_sharded_gn_v2.cpp"
                         : "ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/dataflow/"
                           "reader_mcast_receiver_unary_sharded_gn_v2.cpp");
        reader_mcast_receiver_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
        reader_mcast_receiver_desc.core_ranges = mcast_receiver_cores;
        reader_mcast_receiver_desc.compile_time_args = reader_mcast_receiver_compile_time_args;
        reader_mcast_receiver_desc.defines =
            KernelDescriptor::Defines(reader_mcast_receiver_defines.begin(), reader_mcast_receiver_defines.end());
        reader_mcast_receiver_desc.config = DataMovementConfigDescriptor{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = reader_noc,
        };
    }

    // writer defines
    std::map<std::string, std::string> writer_defines;
    writer_defines["TILE_HW_VAL"] = std::to_string(tile_hw);
    if (negative_mask.has_value()) {
        writer_defines["FUSE_NEGATIVE_MASK"] = "1";
    }
    // writer compile time args
    std::vector<uint32_t> writer_mcast_sender_compile_time_args = {
        1,
        static_cast<uint32_t>(gamma.has_value()),
        static_cast<uint32_t>(beta.has_value()),
        gamma_beta_num_cols_tile_per_core,
        per_core_N,
        per_core_N * datum_size_bytes,
        per_core_Nt * tile_width * datum_size_bytes,
        num_groups_per_core,
        num_batches_per_core,
        block_wt};

    if (gamma.has_value() && gamma.value().layout() == Layout::ROW_MAJOR) {
        auto gamma_stick_size = gamma.value().padded_shape()[3] * gamma.value().element_size();
        writer_mcast_sender_compile_time_args.push_back(gamma_stick_size);
    } else if (beta.has_value() && beta.value().layout() == Layout::ROW_MAJOR) {
        auto beta_stick_size = beta.value().padded_shape()[3] * beta.value().element_size();
        writer_mcast_sender_compile_time_args.push_back(beta_stick_size);
    } else {
        writer_mcast_sender_compile_time_args.push_back(tile_hw * datum_size_bytes);
    }
    writer_mcast_sender_compile_time_args.push_back(
        num_rows_per_batch_per_core * num_datum_row_per_group);                                  // reduce_factor_w
    writer_mcast_sender_compile_time_args.push_back(num_cores_per_batch * num_cores_per_group);  // reduce_factor_c

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

    KernelDescriptor writer_desc;
    writer_desc.kernel_source = writer_kernel;
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = writer_mcast_sender_compile_time_args;
    writer_desc.defines = KernelDescriptor::Defines(writer_defines.begin(), writer_defines.end());
    writer_desc.config = DataMovementConfigDescriptor{
        .processor = DataMovementProcessor::RISCV_1,
        .noc = writer_noc,
    };

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
        1,
        static_cast<uint32_t>(gamma.has_value()),
        static_cast<uint32_t>(beta.has_value()),
        num_cores_per_mcast_group,
        num_batches_per_core,
        num_groups_per_core,

        num_datum_row_per_group_mod_tile_w,

        block_ht,
        block_wt,
        block_ht * block_wt,

        subblock_wt,
        num_subblocks_w,

        per_core_Mt,
        per_core_Nt,
        per_core_Mt * per_core_Nt,

        per_core_Nt * tile_hw * datum_size_bytes,  // per_core_N_tile_bytes
        num_groups_per_reset,
        single_tile_size,
        per_core_Mt * per_core_Nt / num_batches_per_core,
        num_groups_per_core * block_wt,
        block_wt_last,
        static_cast<uint32_t>((num_datum_row_per_group_mod_tile_w & (num_datum_row_per_group_mod_tile_w - 1)) == 0),
        static_cast<uint32_t>(num_datum_row_per_group < tile_width),
        num_datum_row_per_group - ((block_wt - 1) * tile_width)};
    if (use_welford) {
        mcast_sender_compute_compile_time_args.push_back(num_datum_row_per_group);  // num_cols_per_group
    }
    mcast_sender_compute_compile_time_args.push_back(tile_width);
    std::vector<uint32_t> mcast_receiver_compute_compile_time_args = {
        0,
        static_cast<uint32_t>(gamma.has_value()),
        static_cast<uint32_t>(beta.has_value()),
        num_cores_per_mcast_group,
        num_batches_per_core,
        num_groups_per_core,

        num_datum_row_per_group_mod_tile_w,

        block_ht,
        block_wt,
        block_ht * block_wt,

        subblock_wt,
        num_subblocks_w,

        per_core_Mt,
        per_core_Nt,
        per_core_Mt * per_core_Nt,

        per_core_Nt * tile_hw * datum_size_bytes,  // per_core_N_tile_bytes
        num_groups_per_reset,
        single_tile_size,
        per_core_Mt * per_core_Nt / num_batches_per_core,
        num_groups_per_core * block_wt,
        block_wt_last,
        static_cast<uint32_t>((num_datum_row_per_group_mod_tile_w & (num_datum_row_per_group_mod_tile_w - 1)) == 0),
        static_cast<uint32_t>(num_datum_row_per_group < tile_width),
        num_datum_row_per_group - ((block_wt - 1) * tile_width)};
    if (use_welford) {
        mcast_receiver_compute_compile_time_args.push_back(num_datum_row_per_group);  // num_cols_per_group
    }
    mcast_receiver_compute_compile_time_args.push_back(tile_width);
    // compute kernel
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);
    eltwise_binary_defines["FP32_DEST_ACC"] = fp32_dest_acc_en ? "true" : "false";

    // Float32 input on the welford path requires fp32_dest_acc_en=true as a prerequisite for
    // UnpackToDestFp32 (set below). UnpackToDestFp32 is what bypasses the unpacker's
    // Float32 → TF32 truncation in SrcA; fp32_dest_acc_en provides the 32-bit DEST that
    // UnpackToDestFp32 writes into. Without fp32 DEST, UnpackToDestFp32 can't be enabled
    // and inputs are silently truncated to TF32 (10 mantissa bits) on the way through SrcA.
    TT_FATAL(
        !(use_welford && in_data_format == tt::DataFormat::Float32 && !fp32_dest_acc_en),
        "group_norm welford with Float32 input requires fp32_dest_acc_en=true in the compute "
        "kernel config; otherwise precision is silently lost in the unpacker format conversion.");

    // UnpackToDestFp32 only helps for CBs whose only consumer is an op that supports the
    // unpack-to-DEST path (copy_tile or transpose_tile in fp32 mode).
    // The welford_groupnorm_sharded_v2 kernel feeds both c_0 (non-TILIZE_IN) and c_1
    // (TILIZE_IN) through both transpose_tile (welford intake) and sub_tiles_bcast_scalar
    // (final (x - mean) normalization). The FPU consumer means neither CB can carry the flag
    // directly. The workaround is the multi-buffer-index aliasing pattern: register
    // c_29 as a second buffer index on c_0's SRAM and c_31 on c_1's, set
    // UnpackToDestFp32 on the alias indices, and have the kernel read the welford intake
    // transpose via the alias (UnpackToDest fp32 path preserves the full 23-bit mantissa into
    // DEST for the SFPU welford) while keeping the final-stage sub_tiles_bcast_scalar on the
    // primary index (Default mode, SrcA path).
    //
    // Other FP32 CBs were considered and rejected because, even though they pass through an
    // unpack-to-DEST-capable op, the next consumer is a pack into a CB whose downstream
    // reader is an FPU op reading via SrcA, which truncates to TF32 regardless of what
    // was preserved in DEST. Setting the flag would incur the cost without improving precision:
    //   - cb_xmm (c_2): the (x - mean) intermediate. copy_tile into DEST then pack to cb_x;
    //     cb_x is read by add_tiles (FPU on SrcA) for accumulation.
    //   - cb_x (c_13): accumulates (x - mean) results across groups via repeated add_tiles,
    //     each of which reads cb_x via SrcA (truncating to TF32) before producing the next
    //     FP32 sum. The final stored value does carry one add_tiles step's worth of FP32
    //     precision, so an UnpackToDestFp32 alias on the final copy_tile would
    //     preserve ~ one mantissa-bit step beyond TF32, but the accumulated TF32
    //     errors from previous iteration dominate, so the gain doesn't justify the overhead.
    const bool welford_fp32_alias = use_welford && fp32_dest_acc_en && in_data_format == tt::DataFormat::Float32;
    std::vector<tt::tt_metal::UnpackToDestMode> unpack_to_dest_mode(
        NUM_CIRCULAR_BUFFERS, tt::tt_metal::UnpackToDestMode::Default);
    if (welford_fp32_alias) {
        unpack_to_dest_mode[static_cast<uint32_t>(tt::CBIndex::c_29)] =
            tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
        unpack_to_dest_mode[static_cast<uint32_t>(tt::CBIndex::c_31)] =
            tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
    }

    // Welford-fp32 alias args. Only attached on the welford compute kernel; the non-welford
    // groupnorm_sharded_v2.cpp never references these names. Read by welford_groupnorm_sharded_v2.cpp.
    const uint32_t cb_in0_welford_arg =
        welford_fp32_alias ? static_cast<uint32_t>(tt::CBIndex::c_29) : static_cast<uint32_t>(tt::CBIndex::c_0);
    const uint32_t cb_in_welford_arg =
        welford_fp32_alias ? static_cast<uint32_t>(tt::CBIndex::c_31) : static_cast<uint32_t>(tt::CBIndex::c_1);
    KernelDescriptor::NamedCompileTimeArgs welford_named_compile_time_args;
    if (use_welford) {
        welford_named_compile_time_args = {
            {"welford_fp32_alias", static_cast<uint32_t>(welford_fp32_alias)},
            {"cb_in0_welford", cb_in0_welford_arg},
            {"cb_in_welford", cb_in_welford_arg},
        };
    }

    KernelDescriptor compute_sender_desc;
    compute_sender_desc.kernel_source =
        (use_welford
             ? "ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/compute/"
               "welford_groupnorm_sharded_v2.cpp"
             : "ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/compute/groupnorm_sharded_v2.cpp");
    compute_sender_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_sender_desc.core_ranges = mcast_sender_cores;
    compute_sender_desc.compile_time_args = mcast_sender_compute_compile_time_args;
    compute_sender_desc.named_compile_time_args = welford_named_compile_time_args;
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
    compute_receiver_desc.kernel_source =
        (use_welford
             ? "ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/compute/"
               "welford_groupnorm_sharded_v2.cpp"
             : "ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/compute/groupnorm_sharded_v2.cpp");
    compute_receiver_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_receiver_desc.core_ranges = mcast_receiver_cores;
    compute_receiver_desc.compile_time_args = mcast_receiver_compute_compile_time_args;
    compute_receiver_desc.named_compile_time_args = std::move(welford_named_compile_time_args);
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
    constexpr uint32_t in0_welford_alias_index = tt::CBIndex::c_29;
    constexpr uint32_t output_cb_index = tt::CBIndex::c_16;
    uint32_t in0_cb_page_size = reader_repack_output ? a.buffer()->page_size() : in_single_tile_size;
    if (inplace) {
        // input and output share the same CB and globally allocated buffer
        CBDescriptor in0_desc{
            .total_size = in0_CB_size,
            .core_ranges = all_cores,
            .format_descriptors =
                {{CBFormatDescriptor{
                      .buffer_index = static_cast<uint8_t>(in0_cb_index),
                      .data_format = in_data_format,
                      .page_size = in0_cb_page_size,
                  },
                  CBFormatDescriptor{
                      .buffer_index = static_cast<uint8_t>(output_cb_index),
                      .data_format = in_data_format,
                      .page_size = in0_cb_page_size,
                  }}},
            .buffer = a.buffer(),
        };
        if (welford_fp32_alias) {
            in0_desc.format_descriptors.push_back(CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(in0_welford_alias_index),
                .data_format = in_data_format,
                .page_size = in0_cb_page_size,
            });
        }
        desc.cbs.push_back(std::move(in0_desc));
    } else {
        CBDescriptor in0_desc{
            .total_size = in0_CB_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(in0_cb_index),
                .data_format = in_data_format,
                .page_size = in0_cb_page_size,
            }}},
            .buffer = a.buffer(),
        };
        if (welford_fp32_alias) {
            in0_desc.format_descriptors.push_back(CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(in0_welford_alias_index),
                .data_format = in_data_format,
                .page_size = in0_cb_page_size,
            });
        }
        desc.cbs.push_back(std::move(in0_desc));

        uint32_t out_cb_total_size = reader_repack_output ? output.buffer()->aligned_size_per_bank() : out_CB_size;
        uint32_t out_cb_page_size = reader_repack_output ? output.buffer()->page_size() : out_single_tile_size;
        desc.cbs.push_back(CBDescriptor{
            .total_size = out_cb_total_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(output_cb_index),
                .data_format = out_data_format,
                .page_size = out_cb_page_size,
            }}},
            .buffer = output.buffer(),
        });
    }

    constexpr uint32_t in_welford_alias_index = tt::CBIndex::c_31;
    auto make_in_cb_desc = [&](uint32_t total_size) {
        CBDescriptor in_desc{
            .total_size = total_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_1),
                .data_format = in_data_format,
                .page_size = in_single_tile_size,
            }}},
        };
        if (welford_fp32_alias) {
            in_desc.format_descriptors.push_back(CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(in_welford_alias_index),
                .data_format = in_data_format,
                .page_size = in_single_tile_size,
            });
        }
        return in_desc;
    };
    if (!negative_mask.has_value()) {
        // in - stores tilized input
        desc.cbs.push_back(make_in_cb_desc(in_CB_size));
        if (untilize_out) {
            constexpr uint32_t out_cb_index = tt::CBIndex::c_30;
            desc.cbs.push_back(CBDescriptor{
                .total_size = in_CB_size,
                .core_ranges = all_cores,
                .format_descriptors = {{CBFormatDescriptor{
                    .buffer_index = static_cast<uint8_t>(out_cb_index),
                    .data_format = in_data_format,
                    .page_size = in_single_tile_size,
                }}},
            });
        }
    } else {
        // in - stores tilized input
        // tilized in is overlapped with it
        desc.cbs.push_back(make_in_cb_desc(in_CB_size));
    }
    // in2 scaler - for partial Ex
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
    // in3 eps
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
    // in4 scaler-c
    if (!use_welford) {
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
    }
    // gamma
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
    // beta
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
    // input mask
    if (input_mask.has_value()) {
        constexpr uint32_t in_mask_cb_index = tt::CBIndex::c_7;
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
    // negative mask
    if (negative_mask.has_value()) {
        constexpr uint32_t in_negative_mask_cb_index = tt::CBIndex::c_14;
        desc.cbs.push_back(CBDescriptor{
            .total_size = in_negative_mask_CB_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(in_negative_mask_cb_index),
                .data_format = in_negative_mask_cb_data_format,
                .page_size = in_negative_mask_single_tile_size,
            }}},
        });
    }
    if (reader_repack_output) {
        constexpr uint32_t repack_cb_index = tt::CBIndex::c_11;
        constexpr uint32_t repack_out_cb_index = tt::CBIndex::c_12;
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
    // x
    constexpr uint32_t x_cb_index = tt::CBIndex::c_13;
    desc.cbs.push_back(CBDescriptor{
        .total_size = x_CB_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(x_cb_index),
            .data_format = cb_data_format,
            .page_size = single_tile_size,
        }}},
    });

    // ex_partial
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

    // ex_external: Not used by Welford.
    if (!use_welford) {
        constexpr uint32_t ex_cb_external_index = tt::CBIndex::c_10;
        desc.cbs.push_back(CBDescriptor{
            .total_size = single_tile_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(ex_cb_external_index),
                .data_format = cb_data_format,
                .page_size = single_tile_size,
            }}},
        });
    }

    // ex_global
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

    // ex2pe
    constexpr uint32_t cb_ex2pe_index = tt::CBIndex::c_17;
    desc.cbs.push_back(CBDescriptor{
        .total_size = ex2pe_CB_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_ex2pe_index),
            .data_format = cb_data_format,
            .page_size = single_tile_size,
        }}},
    });

    constexpr uint32_t cb_ones_index = tt::CBIndex::c_26;
    desc.cbs.push_back(CBDescriptor{
        .total_size = single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_ones_index),
            .data_format = cb_data_format,
            .page_size = single_tile_size,
        }}},
    });

    // Runtime Args
    uint32_t eps_u = std::bit_cast<uint32_t>(eps);

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
                mcast_sender_args.push_back(static_cast<uint32_t>(!mcast_group_first.empty()));
                mcast_sender_args.push_back(static_cast<uint32_t>(!mcast_group_last.empty()));
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
                log_debug(tt::LogOp, "mcast receiver receive from coord: {} {}", group.front().x, group.front().y);
                std::vector<uint32_t> mcast_receiver_args;
                mcast_receiver_args.push_back(device->worker_core_from_logical_core(group.front()).x);
                mcast_receiver_args.push_back(device->worker_core_from_logical_core(group.front()).y);
                reader_mcast_receiver_desc.runtime_args.emplace_back(core, std::move(mcast_receiver_args));
            }
        }
    }

    // writer
    uint32_t gamma_tile_start_id = 0;
    uint32_t beta_tile_start_id = 0;
    uint32_t input_mask_tile_start_id = 0;
    for (const auto& core : core_coords) {
        tt::tt_metal::KernelDescriptor::RTArgList writer_mcast_sender_args;
        writer_mcast_sender_args.push_back(eps_u);
        if (gamma.has_value()) {
            writer_mcast_sender_args.push_back(gamma.value().buffer());
        } else {
            writer_mcast_sender_args.push_back(0u);
        }
        if (beta.has_value()) {
            writer_mcast_sender_args.push_back(beta.value().buffer());
        } else {
            writer_mcast_sender_args.push_back(0u);
        }
        if (input_mask.has_value()) {
            writer_mcast_sender_args.push_back(input_mask.value().buffer());
        } else {
            writer_mcast_sender_args.push_back(0u);
        }
        if (negative_mask.has_value()) {
            writer_mcast_sender_args.push_back(negative_mask.value().buffer());
        } else {
            writer_mcast_sender_args.push_back(0u);
        }
        writer_mcast_sender_args.push_back(gamma_tile_start_id);
        writer_mcast_sender_args.push_back(beta_tile_start_id);
        writer_mcast_sender_args.push_back(input_mask_tile_start_id);
        writer_desc.emplace_runtime_args(core, writer_mcast_sender_args);

        if (gamma.has_value()) {
            gamma_tile_start_id = (gamma_tile_start_id + gamma_beta_num_cols_tile_per_core) %
                                  (gamma.value().physical_volume() / tile_width);
        }
        if (beta.has_value()) {
            beta_tile_start_id = (beta_tile_start_id + gamma_beta_num_cols_tile_per_core) %
                                 (beta.value().physical_volume() / tile_width);
        }
        if (input_mask.has_value()) {
            // Tile id for negative mask is same as input mask
            input_mask_tile_start_id = (input_mask_tile_start_id + input_mask_num_tiles_per_core) %
                                       (input_mask.value().physical_volume() / tile_hw);
        }
    }

    desc.kernels.push_back(std::move(reader_mcast_sender_desc));
    if (has_receiver_kernel) {
        desc.kernels.push_back(std::move(reader_mcast_receiver_desc));
    }
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_sender_desc));
    desc.kernels.push_back(std::move(compute_receiver_desc));

    return desc;
}

}  // namespace ttnn::prim

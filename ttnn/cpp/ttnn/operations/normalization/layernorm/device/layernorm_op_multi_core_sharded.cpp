// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <string>

#include "ttnn/operations/normalization/layernorm/device/layernorm_op_multi_core_sharded.hpp"
#include <tt-metalium/circular_buffer_config.hpp>
#include "ttnn/operations/normalization/layernorm/device/layernorm_common.hpp"
#include "ttnn/operations/normalization/layernorm/device/layernorm_device_operation_types.hpp"
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/math.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include <optional>
#include <bit>

using uint32_t = std::uint32_t;
using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

void assert_subblock_compute_config_compatible(bool dst_full_sync_en, bool fp32_dest_acc_en, uint32_t subblock_wt) {
    if (!dst_full_sync_en) {
        if (fp32_dest_acc_en) {
            TT_FATAL(
                subblock_wt <= 4,
                "subblock_wt={}, but subblock width must less than 4 tiles in fp32 mode when dst_full_sync_en is false",
                subblock_wt);
        } else {
            TT_FATAL(
                subblock_wt <= 8,
                "subblock_wt={}, but subblock width must less than 8 tiles when dst_full_sync_en is false",
                subblock_wt);
        }
    } else {
        if (fp32_dest_acc_en) {
            TT_FATAL(
                subblock_wt <= 8,
                "subblock_wt={}, but subblock width must less than 8 tiles in fp32 mode when dst_full_sync_en is true",
                subblock_wt);
        } else {
            TT_FATAL(
                subblock_wt <= 16,
                "subblock_wt={}, but subblock width must less than 16 tiles when dst_full_sync_en is true",
                subblock_wt);
        }
    }
}

std::tuple<tt::DataFormat, tt::DataFormat, tt::DataFormat, tt::DataFormat, tt::DataFormat> get_cb_data_formats(
    const Tensor& output,
    const std::optional<const Tensor>& gamma,
    const std::optional<const Tensor>& beta,
    bool fp32_dest_acc_en) {
    tt::DataFormat out_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    tt::DataFormat cb_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    tt::DataFormat gamma_cb_data_format = gamma.has_value()
                                              ? tt::tt_metal::datatype_to_dataformat_converter(gamma.value().dtype())
                                              : tt::DataFormat::Float16_b;
    tt::DataFormat beta_cb_data_format = beta.has_value()
                                             ? tt::tt_metal::datatype_to_dataformat_converter(beta.value().dtype())
                                             : tt::DataFormat::Float16_b;
    tt::DataFormat reciprocal_cb_data_format = tt::DataFormat::Float32;
    return {out_data_format, cb_data_format, gamma_cb_data_format, beta_cb_data_format, reciprocal_cb_data_format};
}

bool should_use_two_stage_reduce(
    bool mcast_1d, bool row_wise, CoreCoord grid_size, CoreCoord compute_with_storage_grid_size) {
    if (mcast_1d) {
        // only do this for row/col dim are full length
        // row major with multiple rows, or col major with multiple cols
        if ((row_wise && grid_size.x > 1 && grid_size.x <= compute_with_storage_grid_size.x && grid_size.y > 1) ||
            (!row_wise && grid_size.x > 1 && grid_size.y == compute_with_storage_grid_size.y)) {
            return true;
        }
    }
    return false;
}

uint32_t get_num_blocks(bool mcast_1d, bool row_wise, CoreCoord grid_size, const ShardSpec& shard_spec) {
    if (mcast_1d) {
        return shard_spec.num_cores();
    }
    if (row_wise) {
        return grid_size.x;
    }
    return grid_size.y;

    return uint32_t{};
}
}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

LayerNormShardedProgramFactory::cached_program_t LayerNormShardedProgramFactory::create(
    const LayerNormParams& operation_attributes, const LayerNormInputs& tensor_args, Tensor& tensor_return_value) {
    using namespace CMAKE_UNIQUE_NAMESPACE;

    // Extract from operation_attributes and tensor_args
    const auto& a = tensor_args.input;
    const auto& b = tensor_args.residual_input_tensor;
    const auto& gamma = tensor_args.weight;
    const auto& beta = tensor_args.bias;
    const auto& stats = tensor_args.stats;
    auto& output = tensor_return_value;
    bool rms_norm = operation_attributes.norm_type == LayerNormType::RMSNORM;
    bool is_pre_all_gather = operation_attributes.distributed_norm_stage == DistributedLayerNormStage::PRE_ALL_GATHER;
    bool is_post_all_gather = operation_attributes.distributed_norm_stage == DistributedLayerNormStage::POST_ALL_GATHER;
    float eps = operation_attributes.eps;
    const auto& compute_kernel_config = operation_attributes.compute_kernel_config;

    // Extract program config
    CoreCoord compute_with_storage_grid_size;
    uint32_t subblock_wt = 0;
    uint32_t block_ht = 0;
    uint32_t block_wt = 0;
    bool legacy_reduction = false;
    bool legacy_rsqrt = false;
    bool use_welford = false;
    std::visit(
        [&](const auto& program_config) {
            using ProgramConfigType = std::decay_t<decltype(program_config)>;
            if constexpr (std::is_same_v<ProgramConfigType, LayerNormShardedMultiCoreProgramConfig>) {
                compute_with_storage_grid_size = program_config.compute_with_storage_grid_size;
                subblock_wt = program_config.subblock_w;
                block_ht = program_config.block_h;
                block_wt = program_config.block_w;
                legacy_reduction = program_config.legacy_reduction;
                legacy_rsqrt = program_config.legacy_rsqrt;
                use_welford = program_config.use_welford;
            }
        },
        operation_attributes.program_config);

    uint32_t block_wt_resharded = output.shard_spec().value().shape[1] / TILE_WIDTH;
    bool skip_write_back = output.shard_spec().value() == a.shard_spec().value();

    ////////////////////////////////////////////////////////////////////////////
    //                            Device Setup
    ////////////////////////////////////////////////////////////////////////////
    IDevice* device = a.device();

    // convert data format
    tt::DataFormat in_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);

    assert_subblock_compute_config_compatible(dst_full_sync_en, fp32_dest_acc_en, subblock_wt);

    auto [out_data_format, cb_data_format, gamma_cb_data_format, beta_cb_data_format, reciprocal_cb_data_format] =
        get_cb_data_formats(output, gamma, beta, fp32_dest_acc_en);

    // tile sizes
    uint32_t in_single_tile_size = tt::tile_size(in_data_format);
    uint32_t single_tile_size = tt::tile_size(cb_data_format);
    uint32_t out_single_tile_size = tt::tile_size(out_data_format);
    uint32_t gamma_single_tile_size = tt::tile_size(gamma_cb_data_format);
    uint32_t beta_single_tile_size = tt::tile_size(beta_cb_data_format);
    uint32_t bfloat16_tile_size = tt::tile_size(tt::DataFormat::Float16_b);

    log_debug(tt::LogOp, "in_data_format: {}", in_data_format);
    log_debug(tt::LogOp, "out_data_format: {}", out_data_format);
    log_debug(tt::LogOp, "cb_data_format: {}", cb_data_format);
    log_debug(tt::LogOp, "gamma_cb_data_format: {}", gamma_cb_data_format);
    log_debug(tt::LogOp, "beta_cb_data_format: {}", beta_cb_data_format);
    log_debug(tt::LogOp, "reciprocal_cb_data_format: {}", reciprocal_cb_data_format);
    log_debug(tt::LogOp, "math_fidelity: {}", math_fidelity);
    log_debug(tt::LogOp, "math_approx_mode: {}", math_approx_mode);
    log_debug(tt::LogOp, "fp32_dest_acc_en: {}", fp32_dest_acc_en);

    // tensor shape
    const auto& shape = a.padded_shape();
    uint32_t M = a.physical_volume() / shape[-1];
    uint32_t K = shape[-1];
    uint32_t Kt = K / TILE_WIDTH;
    // block
    uint32_t block_w = block_wt * TILE_WIDTH;
    uint32_t block_h = block_ht * TILE_HEIGHT;
    ShardSpec shard_spec = a.shard_spec().value();

    bool mcast_1d = M == block_h;
    bool row_wise = shard_spec.orientation == ShardOrientation::ROW_MAJOR;
    auto bbox = shard_spec.grid.bounding_box();
    CoreCoord grid_size = {bbox.end_coord.x - bbox.start_coord.x + 1, bbox.end_coord.y - bbox.start_coord.y + 1};
    std::optional<CoreCoord> grid_offset = std::nullopt;
    if (bbox.start_coord.x != 0 || bbox.start_coord.y != 0) {
        grid_offset = bbox.start_coord;
    }

    uint32_t num_blocks = get_num_blocks(mcast_1d, row_wise, grid_size, shard_spec);

    // two-stage reduce
    bool use_two_stage_reduce =
        should_use_two_stage_reduce(mcast_1d, row_wise, grid_size, device->compute_with_storage_grid_size());

    uint32_t num_subblocks_w = block_wt / subblock_wt;

    // Get all storage cores
    ShardSpec output_shard_spec = output.shard_spec().value();
    bool output_row_wise = output_shard_spec.orientation == ShardOrientation::ROW_MAJOR;

    CoreRangeSet all_storage_cores = output_shard_spec.grid;
    CoreRangeSet all_worker_and_storage_cores = all_storage_cores.merge(a.shard_spec().value().grid);
    std::vector<uint32_t> storage_core_noc_x;
    std::vector<uint32_t> storage_core_noc_y;
    std::vector<CoreCoord> storage_core_coords =
        corerange_to_cores(all_storage_cores, all_storage_cores.num_cores(), row_wise = output_row_wise);
    for (auto core : storage_core_coords) {
        storage_core_noc_x.push_back((std::uint32_t)device->worker_core_from_logical_core(core).x);
        storage_core_noc_y.push_back((std::uint32_t)device->worker_core_from_logical_core(core).y);

        log_debug(
            tt::LogOp,
            "Storage core: ({}, {}), physical coords: ({}, {})",
            core.x,
            core.y,
            storage_core_noc_x.back(),
            storage_core_noc_y.back());
    }

    // get sharded addr
    // b, gamma, beta addr
    auto gamma_dram_addr = gamma.has_value() ? gamma.value().buffer()->address() : 0;
    auto beta_dram_addr = beta.has_value() ? beta.value().buffer()->address() : 0;
    // num tiles for a, gamma, beta

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    // block size for in0 (tensor a)
    uint32_t num_rows_per_all_to_all_worker = tt::div_up(block_ht, num_blocks);
    if (use_two_stage_reduce) {
        if (row_wise) {
            num_rows_per_all_to_all_worker = tt::div_up(block_ht, grid_size.x);
        } else {
            num_rows_per_all_to_all_worker = tt::div_up(block_ht, grid_size.y);
        }
    }
    uint32_t num_rows_per_all_to_all_worker_last =
        block_ht - ((block_ht / num_rows_per_all_to_all_worker) * num_rows_per_all_to_all_worker);
    uint32_t in0_block_tiles = block_wt * block_ht;
    // pre_all_gather_stats_block_tiles
    uint32_t pre_all_gather_stats_block_tiles = rms_norm ? 1 : 2;
    // post_all_gather_stats_block_tiles
    uint32_t post_all_gather_stats_block_tiles = 1;
    uint32_t num_distributed_devices = 1;
    if (is_post_all_gather && stats.has_value()) {
        post_all_gather_stats_block_tiles = stats.value().padded_shape()[-1] / TILE_WIDTH;
        num_distributed_devices = post_all_gather_stats_block_tiles / pre_all_gather_stats_block_tiles;
    }

    uint32_t in0_CB_tiles = in0_block_tiles;
    uint32_t in0_CB_size = in0_CB_tiles * in_single_tile_size;
    // block size for in1 (tensor b)
    uint32_t in1_CB_size = in0_CB_size;
    // in2 - scaler
    uint32_t in2_CB_size = bfloat16_tile_size;
    // in3 - eps
    uint32_t in3_CB_size = bfloat16_tile_size;
    // gamma
    uint32_t in5_CB_size = in0_block_tiles * gamma_single_tile_size / block_ht;
    // beta
    uint32_t in6_CB_size = in0_block_tiles * beta_single_tile_size / block_ht;
    // itermediate buffers change later
    uint32_t x_CB_size = in0_block_tiles * single_tile_size;
    uint32_t xmm_CB_size = in0_block_tiles * single_tile_size;
    uint32_t ex_partial_CB_size = in0_block_tiles * single_tile_size / block_wt;
    uint32_t ex_external_CB_size = tt::div_up(Kt, block_wt) * single_tile_size;
    if (is_pre_all_gather || is_post_all_gather) {
        ex_partial_CB_size = ex_partial_CB_size * pre_all_gather_stats_block_tiles;
    }
    uint32_t ex_CB_size = ex_partial_CB_size;
    uint32_t ex_global_CB_size = ex_partial_CB_size;
    uint32_t ex2pe_CB_size = num_rows_per_all_to_all_worker * single_tile_size;
    uint32_t stats_cb_size = 0;
    uint32_t stats_reduced_cb_size = 0;
    if (is_post_all_gather) {
        stats_cb_size = post_all_gather_stats_block_tiles * single_tile_size;
        stats_reduced_cb_size = pre_all_gather_stats_block_tiles * single_tile_size;
    }
    // output buffer size
    uint32_t out_CB_size;
    if (is_pre_all_gather) {
        out_CB_size = pre_all_gather_stats_block_tiles * out_single_tile_size;
    } else {
        out_CB_size = in0_block_tiles * out_single_tile_size;
    }
    uint32_t out_reshard_CB_size = out_CB_size;
    if (is_post_all_gather && !skip_write_back) {
        out_reshard_CB_size = block_wt_resharded * block_ht * out_single_tile_size;
    }
    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    Program program = Program();
    // define core ranges
    bool use_mcast = num_blocks > 1;

    uint32_t num_cores_x = grid_size.x;
    uint32_t num_cores_y = grid_size.y;
    uint32_t num_cores_all_to_all = tt::div_up(block_ht, num_rows_per_all_to_all_worker);
    uint32_t num_cores_all_to_all_first_stage = num_cores_all_to_all;
    uint32_t num_cores_all_to_all_second_stage = 0;
    uint32_t num_blocks_first_stage = num_blocks;
    uint32_t num_blocks_second_stage = 0;
    if (use_two_stage_reduce) {
        if (row_wise) {
            num_blocks_first_stage = num_cores_x;
            num_cores_all_to_all_second_stage = num_cores_y;
            num_cores_all_to_all *= num_cores_y;
        } else {
            num_blocks_first_stage = num_cores_y;
            num_cores_all_to_all_second_stage = num_cores_x;
            num_cores_all_to_all *= num_cores_x;
        }
        num_blocks_second_stage = num_cores_all_to_all_second_stage;
    }
    // change tt::CBIndex external size
    if (use_two_stage_reduce) {
        ex_external_CB_size = (num_blocks_first_stage + num_blocks_second_stage - 1) * single_tile_size;
    }
    if (is_pre_all_gather) {
        ex_external_CB_size = ex_external_CB_size * pre_all_gather_stats_block_tiles;
    }

    if (use_welford) {
        // Welford calculates 1 mean tile and 1 var tile per height tile
        // (for row major) or width tile (for col major)
        ex_external_CB_size *= 2;
        ex_partial_CB_size *= 2;
        ex_CB_size *= 2;
        ex_global_CB_size *= 2;
    }

    uint32_t num_none_all_to_all_workers = num_blocks - num_cores_all_to_all;
    if (num_rows_per_all_to_all_worker_last == 0) {
        num_rows_per_all_to_all_worker_last = num_rows_per_all_to_all_worker;
    }

    CoreCoord start_core = {0, 0};
    CoreRangeSet all_cores = shard_spec.grid;
    CoreRange sender_cores(start_core, start_core);
    CoreRangeSet all_to_all_cores;
    CoreRangeSet all_to_all_workers_except_sender;
    CoreRangeSet not_all_to_all_workers;
    uint32_t num_cores_x_mcast, num_cores_y_mcast;
    if (mcast_1d) {
        sender_cores = {start_core, start_core};
        CoreCoord all_core_grid_size;
        CoreCoord none_core_grid_size;
        if (use_two_stage_reduce) {
            if (row_wise) {
                all_core_grid_size = {num_cores_all_to_all_first_stage, num_cores_y};
                none_core_grid_size = {num_cores_x - num_cores_all_to_all_first_stage, num_cores_y};
            } else {
                all_core_grid_size = {num_cores_x, num_cores_all_to_all_first_stage};
                none_core_grid_size = {num_cores_x, num_cores_y - num_cores_all_to_all_first_stage};
            }
        } else {
            all_core_grid_size = grid_size;
            none_core_grid_size = grid_size;
        }
        all_to_all_cores = num_cores_to_corerangeset(start_core, num_cores_all_to_all, all_core_grid_size, row_wise);
        if (row_wise) {
            if (use_mcast) {
                CoreCoord all_start_core;
                CoreCoord end_core = sender_cores.end_coord;
                if (use_two_stage_reduce) {
                    if (end_core.x == all_core_grid_size.x - 1) {
                        all_start_core = {0, end_core.y + 1};
                    } else {
                        all_start_core = {end_core.x + 1, end_core.y};
                    }
                } else {
                    if (end_core.x == bbox.end_coord.x) {
                        all_start_core = {0, end_core.y + 1};
                    } else {
                        all_start_core = {end_core.x + 1, end_core.y};
                    }
                }
                all_to_all_workers_except_sender =
                    num_cores_to_corerangeset(all_start_core, num_cores_all_to_all - 1, all_core_grid_size, row_wise);
            }
            if (num_none_all_to_all_workers > 0) {
                if (use_two_stage_reduce) {
                    CoreCoord none_start_core = {all_core_grid_size.x, sender_cores.end_coord.y};
                    CoreCoord none_end_core = {num_cores_x - 1, num_cores_y - 1};
                    CoreRange none_core_range = CoreRange(none_start_core, none_end_core);
                    not_all_to_all_workers = CoreRangeSet(none_core_range);
                } else {
                    CoreCoord none_start_core;
                    CoreCoord end_core = (*all_to_all_cores.ranges().rbegin()).end_coord;
                    if (end_core.x == bbox.end_coord.x) {
                        none_start_core = {0, end_core.y + 1};
                    } else {
                        none_start_core = {end_core.x + 1, end_core.y};
                    }
                    not_all_to_all_workers = num_cores_to_corerangeset(
                        none_start_core, num_none_all_to_all_workers, none_core_grid_size, row_wise);
                }
            }
        } else {
            if (use_mcast) {
                CoreCoord all_start_core;
                CoreCoord end_core = sender_cores.end_coord;
                if (use_two_stage_reduce) {
                    if (end_core.y == all_core_grid_size.y - 1) {
                        all_start_core = {end_core.x + 1, 0};
                    } else {
                        all_start_core = {end_core.x, end_core.y + 1};
                    }
                } else {
                    if (end_core.y == bbox.end_coord.y) {
                        all_start_core = {end_core.x + 1, 0};
                    } else {
                        all_start_core = {end_core.x, end_core.y + 1};
                    }
                }
                all_to_all_workers_except_sender = num_cores_to_corerangeset(
                    CoreCoord{start_core.x, start_core.y + 1}, num_cores_all_to_all - 1, all_core_grid_size, row_wise);
            }
            if (num_none_all_to_all_workers > 0) {
                if (use_two_stage_reduce) {
                    CoreCoord none_start_core = {sender_cores.end_coord.x, all_core_grid_size.y};
                    CoreCoord none_end_core = {num_cores_x - 1, num_cores_y - 1};
                    CoreRange none_core_range = CoreRange(none_start_core, none_end_core);
                    not_all_to_all_workers = CoreRangeSet(none_core_range);
                } else {
                    CoreCoord none_start_core;
                    CoreCoord end_core = (*all_to_all_cores.ranges().rbegin()).end_coord;
                    if (end_core.y == bbox.end_coord.y) {
                        none_start_core = {end_core.x + 1, 0};
                    } else {
                        none_start_core = {end_core.x, end_core.y + 1};
                    }
                    not_all_to_all_workers = num_cores_to_corerangeset(
                        none_start_core, num_none_all_to_all_workers, none_core_grid_size, row_wise);
                }
            }
        }
        num_cores_x_mcast = num_cores_x;
        num_cores_y_mcast = num_cores_y;
    } else {
        if (row_wise) {
            sender_cores = {
                {(std::size_t)start_core.x, (std::size_t)start_core.y},
                {(std::size_t)start_core.x, (std::size_t)start_core.y + num_cores_y - 1}};
            all_to_all_cores = CoreRangeSet(CoreRange(
                {(std::size_t)start_core.x, (std::size_t)start_core.y},
                {(std::size_t)start_core.x + num_cores_all_to_all - 1, (std::size_t)start_core.y + num_cores_y - 1}));
            if (use_mcast && num_cores_all_to_all > 1) {
                all_to_all_workers_except_sender = CoreRangeSet(CoreRange(
                    {(std::size_t)start_core.x + 1, (std::size_t)start_core.y},
                    {(std::size_t)start_core.x + num_cores_all_to_all - 1,
                     (std::size_t)start_core.y + num_cores_y - 1}));
            }
            if (num_none_all_to_all_workers > 0) {
                not_all_to_all_workers = CoreRangeSet(CoreRange(
                    {(std::size_t)start_core.x + num_cores_all_to_all, (std::size_t)start_core.y},
                    {(std::size_t)start_core.x + num_cores_x - 1, (std::size_t)start_core.y + num_cores_y - 1}));
            }
            num_cores_x_mcast = num_cores_x;
            num_cores_y_mcast = 1;
        } else {
            sender_cores = {
                {(std::size_t)start_core.x, (std::size_t)start_core.y},
                {(std::size_t)start_core.x + num_cores_x - 1, (std::size_t)start_core.y}};
            all_to_all_cores = CoreRangeSet(CoreRange(
                {(std::size_t)start_core.x, (std::size_t)start_core.y},
                {(std::size_t)start_core.x + num_cores_x - 1, (std::size_t)start_core.y + num_cores_all_to_all - 1}));
            if (use_mcast && num_cores_all_to_all > 1) {
                all_to_all_workers_except_sender = CoreRangeSet(CoreRange(
                    {(std::size_t)start_core.x, (std::size_t)start_core.y + 1},
                    {(std::size_t)start_core.x + num_cores_x - 1,
                     (std::size_t)start_core.y + num_cores_all_to_all - 1}));
            }
            if (num_none_all_to_all_workers > 0) {
                not_all_to_all_workers = CoreRangeSet(CoreRange(
                    {(std::size_t)start_core.x, (std::size_t)start_core.y + num_cores_all_to_all},
                    {(std::size_t)start_core.x + num_cores_x - 1, (std::size_t)start_core.y + num_cores_y - 1}));
            }
            num_cores_x_mcast = 1;
            num_cores_y_mcast = num_cores_y;
        }
    }
    auto applyStartOffset = [](const CoreRangeSet& input_set, const CoreCoord& grid_offset) -> CoreRangeSet {
        if (input_set.empty()) {
            return input_set;
        }

        std::vector<CoreRange> new_ranges;
        new_ranges.reserve(input_set.size());

        for (const CoreRange& range : input_set.ranges()) {
            CoreCoord new_start = {range.start_coord.x + grid_offset.x, range.start_coord.y + grid_offset.y};
            CoreCoord new_end = {range.end_coord.x + grid_offset.x, range.end_coord.y + grid_offset.y};
            new_ranges.emplace_back(new_start, new_end);
        }

        return CoreRangeSet(std::move(new_ranges));
    };
    if (grid_offset.has_value()) {
        start_core = {start_core.x + grid_offset.value().x, start_core.y + grid_offset.value().y};
        sender_cores = {
            {sender_cores.start_coord.x + start_core.x, sender_cores.start_coord.y + start_core.y},
            {sender_cores.end_coord.x + start_core.x, sender_cores.end_coord.y + start_core.y}};
        all_to_all_cores = applyStartOffset(all_to_all_cores, grid_offset.value());
        all_to_all_workers_except_sender = applyStartOffset(all_to_all_workers_except_sender, grid_offset.value());
        not_all_to_all_workers = applyStartOffset(not_all_to_all_workers, grid_offset.value());
    }

    // Mcast args
    auto reduce_sender_semaphore_id = tt::tt_metal::CreateSemaphore(program, all_cores, INVALID);
    auto reduce_receiver_semaphore_id = tt::tt_metal::CreateSemaphore(program, all_cores, INVALID);
    auto reduce_second_stage_semaphore_id = tt::tt_metal::CreateSemaphore(program, all_cores, INVALID);
    // reader defines
    std::map<std::string, std::string> reader_mcast_sender_defines;
    std::map<std::string, std::string> reader_mcast_receiver_defines;
    if (b) {
        reader_mcast_sender_defines["FUSE_PRE_ADD"] = "1";
        reader_mcast_receiver_defines["FUSE_PRE_ADD"] = "1";
    }
    if (gamma.has_value()) {
        reader_mcast_sender_defines["FUSE_GAMMA"] = "1";
        reader_mcast_receiver_defines["FUSE_GAMMA"] = "1";
    }
    if (beta.has_value()) {
        reader_mcast_sender_defines["FUSE_BETA"] = "1";
        reader_mcast_receiver_defines["FUSE_BETA"] = "1";
    }
    // reader compile time args
    std::vector<uint32_t> reader_mcast_sender_compile_time_args = {
        (std::uint32_t)reduce_receiver_semaphore_id,
        (std::uint32_t)reduce_sender_semaphore_id,
        (std::uint32_t)num_blocks,
        (std::uint32_t)block_ht,
        (std::uint32_t)block_ht * single_tile_size,
        (std::uint32_t)num_cores_all_to_all_first_stage,
        (std::uint32_t)num_rows_per_all_to_all_worker,
        (std::uint32_t)num_rows_per_all_to_all_worker * single_tile_size,
        (std::uint32_t)num_rows_per_all_to_all_worker_last,
        (std::uint32_t)num_rows_per_all_to_all_worker_last * single_tile_size,
        (std::uint32_t)row_wise,
        (std::uint32_t)num_cores_x_mcast,
        (std::uint32_t)num_cores_y_mcast,
        (std::uint32_t)use_two_stage_reduce,
        (std::uint32_t)num_blocks_first_stage,
        (std::uint32_t)num_blocks_second_stage,
        (std::uint32_t)reduce_second_stage_semaphore_id,
        (std::uint32_t)rms_norm,
        (std::uint32_t)use_welford};
    std::vector<uint32_t> reader_mcast_receiver_all_to_all_compile_time_args = {
        (std::uint32_t)reduce_receiver_semaphore_id,
        (std::uint32_t)reduce_sender_semaphore_id,
        (std::uint32_t)num_blocks,
        (std::uint32_t)block_ht,
        (std::uint32_t)1,
        (std::uint32_t)num_cores_all_to_all_first_stage,
        (std::uint32_t)num_rows_per_all_to_all_worker,
        (std::uint32_t)num_rows_per_all_to_all_worker_last,
        (std::uint32_t)row_wise,
        (std::uint32_t)num_cores_x_mcast,
        (std::uint32_t)num_cores_y_mcast,
        (std::uint32_t)use_two_stage_reduce,
        (std::uint32_t)num_blocks_first_stage,
        (std::uint32_t)num_blocks_second_stage,
        (std::uint32_t)reduce_second_stage_semaphore_id,
        (std::uint32_t)rms_norm,
        (std::uint32_t)use_welford};
    std::vector<uint32_t> reader_mcast_receiver_compile_time_args = {
        (std::uint32_t)reduce_receiver_semaphore_id,
        (std::uint32_t)reduce_sender_semaphore_id,
        (std::uint32_t)num_blocks,
        (std::uint32_t)block_ht,
        (std::uint32_t)0,
        (std::uint32_t)num_cores_all_to_all_first_stage,
        (std::uint32_t)num_rows_per_all_to_all_worker,
        (std::uint32_t)num_rows_per_all_to_all_worker_last,
        (std::uint32_t)row_wise,
        (std::uint32_t)1,
        (std::uint32_t)1,
        (std::uint32_t)0,
        (std::uint32_t)0,
        (std::uint32_t)0,
        (std::uint32_t)reduce_second_stage_semaphore_id,
        (std::uint32_t)rms_norm,
        (std::uint32_t)use_welford};

    tt::tt_metal::NOC reader_noc = tt::tt_metal::detail::preferred_noc_for_dram_read(device->arch());
    tt::tt_metal::NOC writer_noc = tt::tt_metal::detail::preferred_noc_for_dram_write(device->arch());

    if (is_post_all_gather && !skip_write_back) {
        reader_noc = NOC::NOC_0;
        writer_noc = NOC::NOC_1;
    }

    // reader kernel
    std::string sender_reader_kernel_file =
        "ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/dataflow/"
        "reader_mcast_sender_unary_sharded_ln.cpp";
    std::string receiver_reader_kernel_file =
        "ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/dataflow/"
        "reader_mcast_receiver_unary_sharded_ln.cpp";

    if (is_pre_all_gather) {
        sender_reader_kernel_file =
            "ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/dataflow/"
            "reader_mcast_sender_unary_sharded_ln_pre_allgather.cpp";
        receiver_reader_kernel_file =
            "ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/dataflow/"
            "reader_mcast_receiver_unary_sharded_ln_pre_allgather.cpp";
    } else if (is_post_all_gather) {
        sender_reader_kernel_file =
            "ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/dataflow/"
            "reader_mcast_sender_unary_sharded_ln_post_allgather.cpp";
        receiver_reader_kernel_file =
            "ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/dataflow/"
            "reader_mcast_receiver_unary_sharded_ln_post_allgather.cpp";
    }

    auto reader_mcast_sender_kernels_id = CreateKernel(
        program,
        sender_reader_kernel_file,
        sender_cores,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = reader_noc,
            .compile_args = reader_mcast_sender_compile_time_args,
            .defines = reader_mcast_sender_defines});
    KernelHandle reader_mcast_receiver_kernels_id_all_to_all = -1;
    KernelHandle reader_mcast_receiver_kernels_id = -1;
    if (use_mcast) {
        reader_mcast_receiver_kernels_id_all_to_all = CreateKernel(
            program,
            receiver_reader_kernel_file,
            all_to_all_workers_except_sender,
            tt::tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                .noc = reader_noc,
                .compile_args = reader_mcast_receiver_all_to_all_compile_time_args,
                .defines = reader_mcast_receiver_defines});
    }
    if (num_none_all_to_all_workers > 0) {
        reader_mcast_receiver_kernels_id = CreateKernel(
            program,
            receiver_reader_kernel_file,
            not_all_to_all_workers,
            tt::tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                .noc = reader_noc,
                .compile_args = reader_mcast_receiver_compile_time_args,
                .defines = reader_mcast_receiver_defines});
    }

    // writer defines
    std::map<std::string, std::string> writer_defines;
    if (rms_norm) {
        writer_defines["RMSNORM"] = "1";
    }
    if (skip_write_back) {
        writer_defines["SKIP_WRITE_BACK"] = "1";
    }
    // writer compile time args
    std::vector<uint32_t> writer_mcast_sender_compile_time_args = {
        1,  // is_all_to_all_worker
        (std::uint32_t)gamma.has_value(),
        (std::uint32_t)beta.has_value(),
        (std::uint32_t)block_wt,
        (std::uint32_t)use_welford};
    tt::tt_metal::TensorAccessorArgs(gamma ? gamma->buffer() : nullptr)
        .append_to(writer_mcast_sender_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(beta ? beta->buffer() : nullptr).append_to(writer_mcast_sender_compile_time_args);

    std::vector<uint32_t> writer_mcast_receiver_compile_time_args = {
        0,  // is_all_to_all_worker
        (std::uint32_t)gamma.has_value(),
        (std::uint32_t)beta.has_value(),
        (std::uint32_t)block_wt,
        (std::uint32_t)use_welford};
    tt::tt_metal::TensorAccessorArgs(gamma ? gamma->buffer() : nullptr)
        .append_to(writer_mcast_receiver_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(beta ? beta->buffer() : nullptr)
        .append_to(writer_mcast_receiver_compile_time_args);

    if (gamma.has_value() and gamma.value().layout() == Layout::ROW_MAJOR) {
        auto gamma_stick_size = gamma.value().padded_shape()[-1] * gamma.value().element_size();
        writer_mcast_sender_compile_time_args.push_back(gamma_stick_size);
        writer_mcast_receiver_compile_time_args.push_back(gamma_stick_size);
    } else if (beta.has_value() and beta.value().layout() == Layout::ROW_MAJOR) {
        auto beta_stick_size = beta.value().padded_shape()[-1] * beta.value().element_size();
        writer_mcast_sender_compile_time_args.push_back(beta_stick_size);
        writer_mcast_receiver_compile_time_args.push_back(beta_stick_size);
    }
    writer_mcast_sender_compile_time_args.push_back(gamma_cb_data_format == tt::DataFormat::Float32);
    writer_mcast_sender_compile_time_args.push_back(beta_cb_data_format == tt::DataFormat::Float32);
    writer_mcast_receiver_compile_time_args.push_back(gamma_cb_data_format == tt::DataFormat::Float32);
    writer_mcast_receiver_compile_time_args.push_back(beta_cb_data_format == tt::DataFormat::Float32);

    // write back compile time args
    writer_mcast_sender_compile_time_args.push_back(block_wt * out_single_tile_size);  // out_tensor_stride_w_bytes
    writer_mcast_sender_compile_time_args.push_back(
        block_wt_resharded * out_single_tile_size);  // out_reshard_tensor_stride_w_bytes: how many bytes to skip to get
                                                     // to the next data chunk
    writer_mcast_sender_compile_time_args.push_back(block_ht);  // height in tiles

    writer_mcast_receiver_compile_time_args.push_back(block_wt * out_single_tile_size);  // out_tensor_stride_w_bytes
    writer_mcast_receiver_compile_time_args.push_back(
        block_wt_resharded * out_single_tile_size);  // out_reshard_tensor_stride_w_bytes: how many bytes to skip to get
                                                     // to the next data chunk
    writer_mcast_receiver_compile_time_args.push_back(block_ht);  // height in tiles
    writer_mcast_receiver_compile_time_args.push_back(use_welford);

    // writer kernel
    bool use_row_major_kernel = (gamma.has_value() and gamma.value().layout() == Layout::ROW_MAJOR) or
                                (beta.has_value() and beta.value().layout() == Layout::ROW_MAJOR);
    std::string writer_kernel;
    if (is_pre_all_gather) {
        writer_kernel =
            "ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/dataflow/"
            "writer_unary_sharded_ln_pre_all_gather.cpp";
    } else {
        writer_kernel = use_row_major_kernel ? "ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/"
                                               "dataflow/writer_unary_sharded_ln_rm_gb.cpp"
                                             : "ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/"
                                               "dataflow/writer_unary_sharded_ln.cpp";
    }
    auto writer_mcast_sender_kernels_id = CreateKernel(
        program,
        writer_kernel,
        all_to_all_cores,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = writer_noc,
            .compile_args = writer_mcast_sender_compile_time_args,
            .defines = writer_defines});
    KernelHandle writer_mcast_receiver_kernels_id = -1;
    if (num_none_all_to_all_workers > 0) {
        writer_mcast_receiver_kernels_id = CreateKernel(
            program,
            writer_kernel,
            not_all_to_all_workers,
            tt::tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
                .noc = writer_noc,
                .compile_args = writer_mcast_receiver_compile_time_args,
                .defines = writer_defines});
    }
    // defines
    std::map<std::string, std::string> compute_defines;
    if (b) {
        compute_defines["FUSE_PRE_ADD"] = "1";
    }
    if (rms_norm && !use_welford) {
        compute_defines["RMSNORM"] = "1";
    }

    // The last core may not need the entire block_w worth of reciprocals,
    // but the table size has to be a compile-time argument, so we use
    // the max value that it'll be.
    uint32_t per_core_recip_lut_size = block_w;
    auto [recip_tensor, reciprocal_CB_size_bytes] =
        create_reciprocal_tensor_if_needed(device, per_core_recip_lut_size, all_cores, use_welford);

    // compute kernel compile time args
    bool float32_reduction = fp32_dest_acc_en && !legacy_reduction;
    std::vector<uint32_t> all_to_all_except_top_compute_compile_time_args = {
        0,
        gamma.has_value(),
        beta.has_value(),
        num_blocks_first_stage,
        block_ht,
        block_wt,
        subblock_wt,
        num_subblocks_w,
        1,
        block_ht * block_wt,
        fp32_dest_acc_en,
        float32_reduction,
        legacy_rsqrt,
        num_blocks_second_stage};
    std::vector<uint32_t> not_all_to_all_compute_compile_time_args = {
        0,
        gamma.has_value(),
        beta.has_value(),
        num_blocks_first_stage,
        block_ht,
        block_wt,
        subblock_wt,
        num_subblocks_w,
        0,
        block_ht * block_wt,
        fp32_dest_acc_en,
        float32_reduction,
        legacy_rsqrt,
        num_blocks_second_stage};

    constexpr uint32_t tile_width = tt::constants::TILE_WIDTH;
    uint32_t last_tile_W = K - ((K - tile_width) / tile_width) * tile_width;
    union {
        float f;
        uint32_t u;
    } e{};
    e.f = eps;
    if (use_welford) {
        // For Welford combine
        all_to_all_except_top_compute_compile_time_args.push_back(tile_width);
        all_to_all_except_top_compute_compile_time_args.push_back(last_tile_W);
        all_to_all_except_top_compute_compile_time_args.push_back(K);
        all_to_all_except_top_compute_compile_time_args.push_back(e.u);
        all_to_all_except_top_compute_compile_time_args.push_back(per_core_recip_lut_size);
        not_all_to_all_compute_compile_time_args.push_back(tile_width);
        not_all_to_all_compute_compile_time_args.push_back(last_tile_W);
        not_all_to_all_compute_compile_time_args.push_back(K);
        not_all_to_all_compute_compile_time_args.push_back(e.u);
        not_all_to_all_compute_compile_time_args.push_back(per_core_recip_lut_size);
    }

    // compute kernel
    std::string compute_kernel_file;
    if (is_pre_all_gather) {
        compute_kernel_file =
            "ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/compute/"
            "layernorm_sharded_pre_allgather.cpp";
    } else if (is_post_all_gather) {
        compute_kernel_file =
            "ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/compute/"
            "layernorm_sharded_post_allgather.cpp";
    } else {
        compute_kernel_file =
            use_welford
                ? "ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/compute/"
                  "layernorm_sharded_welford.cpp"
                : "ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/compute/layernorm_sharded.cpp";
    }

    KernelHandle compute_kernels_id = -1;
    auto compute_kernels_id_all_to_all = CreateKernel(
        program,
        compute_kernel_file,
        all_to_all_cores,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .math_approx_mode = math_approx_mode,
            .compile_args = all_to_all_except_top_compute_compile_time_args,
            .defines = compute_defines});
    if (num_none_all_to_all_workers > 0) {
        compute_kernels_id = CreateKernel(
            program,
            compute_kernel_file,
            not_all_to_all_workers,
            tt::tt_metal::ComputeConfig{
                .math_fidelity = math_fidelity,
                .fp32_dest_acc_en = fp32_dest_acc_en,
                .math_approx_mode = math_approx_mode,
                .compile_args = not_all_to_all_compute_compile_time_args,
                .defines = compute_defines});
    }
    // Create circular buffers
    // in0 sharded
    uint32_t in0_cb_index = tt::CBIndex::c_0;
    tt::tt_metal::CircularBufferConfig in0_cb_config =
        tt::tt_metal::CircularBufferConfig(in0_CB_size, {{in0_cb_index, in_data_format}})
            .set_page_size(in0_cb_index, in_single_tile_size)
            .set_globally_allocated_address(*a.buffer());
    auto cb_in0 = tt::tt_metal::CreateCircularBuffer(program, all_cores, in0_cb_config);
    // in1 sharded
    uint32_t in1_cb_index = tt::CBIndex::c_1;
    CBHandle cb_in1 = 0;
    CBHandle cb_add_out = 0;
    if (b) {
        tt::tt_metal::CircularBufferConfig in1_cb_config =
            tt::tt_metal::CircularBufferConfig(in1_CB_size, {{in1_cb_index, in_data_format}})
                .set_page_size(in1_cb_index, in_single_tile_size)
                .set_globally_allocated_address(*b.value().buffer());
        cb_in1 = tt::tt_metal::CreateCircularBuffer(program, all_cores, in1_cb_config);
        if (is_pre_all_gather) {
            uint32_t add_out_cb_index = tt::CBIndex::c_14;
            tt::tt_metal::CircularBufferConfig add_out_cb_config =
                tt::tt_metal::CircularBufferConfig(in1_CB_size, {{add_out_cb_index, in_data_format}})
                    .set_page_size(add_out_cb_index, in_single_tile_size)
                    .set_globally_allocated_address(*a.buffer());
            cb_add_out = tt::tt_metal::CreateCircularBuffer(program, all_cores, add_out_cb_config);
        }
    }
    // gamma
    if (gamma.has_value()) {
        uint32_t in5_cb_index = tt::CBIndex::c_5;
        tt::tt_metal::CircularBufferConfig in5_cb_config =
            tt::tt_metal::CircularBufferConfig(in5_CB_size, {{in5_cb_index, gamma_cb_data_format}})
                .set_page_size(in5_cb_index, gamma_single_tile_size);
        tt::tt_metal::CreateCircularBuffer(program, all_cores, in5_cb_config);
    }
    // beta
    if (beta.has_value()) {
        uint32_t in6_cb_index = tt::CBIndex::c_6;
        tt::tt_metal::CircularBufferConfig in6_cb_config =
            tt::tt_metal::CircularBufferConfig(in6_CB_size, {{in6_cb_index, beta_cb_data_format}})
                .set_page_size(in6_cb_index, beta_single_tile_size);
        tt::tt_metal::CreateCircularBuffer(program, all_cores, in6_cb_config);
    }
    // x
    uint32_t x_cb_index;
    x_cb_index = tt::CBIndex::c_24;
    tt::tt_metal::CircularBufferConfig x_cb_config =
        tt::tt_metal::CircularBufferConfig(x_CB_size, {{x_cb_index, cb_data_format}})
            .set_page_size(x_cb_index, single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, x_cb_config);
    // xmm
    uint32_t xmm_cb_index;
    xmm_cb_index = tt::CBIndex::c_18;
    tt::tt_metal::CircularBufferConfig xmm_cb_config =
        tt::tt_metal::CircularBufferConfig(xmm_CB_size, {{xmm_cb_index, cb_data_format}})
            .set_page_size(xmm_cb_index, single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, xmm_cb_config);
    // ex_partial
    if (!rms_norm) {
        uint32_t ex_cb_partial_index = tt::CBIndex::c_8;
        tt::tt_metal::CircularBufferConfig ex_cb_partial_config =
            tt::tt_metal::CircularBufferConfig(ex_partial_CB_size, {{ex_cb_partial_index, cb_data_format}})
                .set_page_size(ex_cb_partial_index, single_tile_size);
        tt::tt_metal::CreateCircularBuffer(program, all_cores, ex_cb_partial_config);
        // ex
        uint32_t ex_cb_index = tt::CBIndex::c_9;
        tt::tt_metal::CircularBufferConfig ex_cb_config =
            tt::tt_metal::CircularBufferConfig(ex_CB_size, {{ex_cb_index, cb_data_format}})
                .set_page_size(ex_cb_index, single_tile_size);
        tt::tt_metal::CreateCircularBuffer(program, all_cores, ex_cb_config);
        // ex_external
        uint32_t ex_cb_external_index = tt::CBIndex::c_10;
        tt::tt_metal::CircularBufferConfig ex_cb_external_config =
            tt::tt_metal::CircularBufferConfig(ex_external_CB_size, {{ex_cb_external_index, cb_data_format}})
                .set_page_size(ex_cb_external_index, single_tile_size);
        tt::tt_metal::CreateCircularBuffer(program, all_cores, ex_cb_external_config);
    }
    if (!use_welford) {
        // in2 scaler
        uint32_t in2_cb_index = tt::CBIndex::c_2;
        tt::tt_metal::CircularBufferConfig in2_cb_config =
            tt::tt_metal::CircularBufferConfig(in2_CB_size, {{in2_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(in2_cb_index, bfloat16_tile_size);
        tt::tt_metal::CreateCircularBuffer(program, all_cores, in2_cb_config);
        // in3 eps
        uint32_t in3_cb_index = tt::CBIndex::c_3;
        tt::tt_metal::CircularBufferConfig in3_cb_config =
            tt::tt_metal::CircularBufferConfig(in3_CB_size, {{in3_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(in3_cb_index, bfloat16_tile_size);
        tt::tt_metal::CreateCircularBuffer(program, all_cores, in3_cb_config);
        // in4 scaler-c
        uint32_t in4_cb_index = tt::CBIndex::c_4;
        tt::tt_metal::CircularBufferConfig in4_cb_config =
            tt::tt_metal::CircularBufferConfig(in2_CB_size, {{in4_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(in4_cb_index, bfloat16_tile_size);
        tt::tt_metal::CreateCircularBuffer(program, all_cores, in4_cb_config);
        // ex_partial2
        uint32_t ex_cb_partial2_index = tt::CBIndex::c_11;
        tt::tt_metal::CircularBufferConfig ex_cb_partial2_config =
            tt::tt_metal::CircularBufferConfig(ex_partial_CB_size, {{ex_cb_partial2_index, cb_data_format}})
                .set_page_size(ex_cb_partial2_index, single_tile_size);
        tt::tt_metal::CreateCircularBuffer(program, all_cores, ex_cb_partial2_config);
        // ex2
        uint32_t ex2_cb_index = tt::CBIndex::c_12;
        tt::tt_metal::CircularBufferConfig ex2_cb_config =
            tt::tt_metal::CircularBufferConfig(ex_CB_size, {{ex2_cb_index, cb_data_format}})
                .set_page_size(ex2_cb_index, single_tile_size);
        tt::tt_metal::CreateCircularBuffer(program, all_cores, ex2_cb_config);
        // ex_external2
        uint32_t ex_cb_external2_index = tt::CBIndex::c_13;
        tt::tt_metal::CircularBufferConfig ex_cb_external2_config =
            tt::tt_metal::CircularBufferConfig(ex_external_CB_size, {{ex_cb_external2_index, cb_data_format}})
                .set_page_size(ex_cb_external2_index, single_tile_size);
        tt::tt_metal::CreateCircularBuffer(program, all_cores, ex_cb_external2_config);
        // ex2pe
        uint32_t cb_ex2pe_index;
        cb_ex2pe_index = tt::CBIndex::c_20;
        tt::tt_metal::CircularBufferConfig ex2pe_cb_config =
            tt::tt_metal::CircularBufferConfig(ex2pe_CB_size, {{cb_ex2pe_index, cb_data_format}})
                .set_page_size(cb_ex2pe_index, single_tile_size);
        tt::tt_metal::CreateCircularBuffer(program, all_cores, ex2pe_cb_config);
    }
    // ex_global
    uint32_t ex_global_cb_index = tt::CBIndex::c_15;
    tt::tt_metal::CircularBufferConfig ex_global_cb_config =
        tt::tt_metal::CircularBufferConfig(ex_global_CB_size, {{ex_global_cb_index, cb_data_format}})
            .set_page_size(ex_global_cb_index, single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, ex_global_cb_config);
    if (use_welford) {
        // Intermediate buffer to store transposed Welford results
        // This is only needed as a workaround for a transpose_wh_dest() bug
        uint32_t cb_transpose_index = tt::CBIndex::c_22;
        tt::tt_metal::CircularBufferConfig cb_transpose_config =
            tt::tt_metal::CircularBufferConfig(ex_global_CB_size, {{cb_transpose_index, cb_data_format}})
                .set_page_size(cb_transpose_index, single_tile_size);
        tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_transpose_config);

        // Reciprocal LUT
        CircularBufferConfig cb_recip_config =
            CircularBufferConfig(reciprocal_CB_size_bytes, {{tt::CBIndex::c_25, reciprocal_cb_data_format}})
                .set_page_size(tt::CBIndex::c_25, reciprocal_CB_size_bytes)
                .set_globally_allocated_address(*recip_tensor.value().buffer());
        CreateCircularBuffer(program, all_cores, cb_recip_config);
    }

    CBHandle cb_stats = 0;
    if (is_post_all_gather) {
        // cb_stats
        uint32_t cb_stats_index;
        cb_stats_index = tt::CBIndex::c_7;
        tt::tt_metal::CircularBufferConfig stats_cb_config =
            tt::tt_metal::CircularBufferConfig(stats_cb_size, {{cb_stats_index, cb_data_format}})
                .set_page_size(cb_stats_index, single_tile_size)
                .set_globally_allocated_address(*stats.value().buffer());
        cb_stats = tt::tt_metal::CreateCircularBuffer(program, sender_cores, stats_cb_config);
        // cb_stats_reduced
        uint32_t cb_stats_reduced_index;
        cb_stats_reduced_index = tt::CBIndex::c_21;
        tt::tt_metal::CircularBufferConfig stats_reduced_cb_config =
            tt::tt_metal::CircularBufferConfig(stats_reduced_cb_size, {{cb_stats_reduced_index, cb_data_format}})
                .set_page_size(cb_stats_reduced_index, single_tile_size);
        tt::tt_metal::CreateCircularBuffer(program, sender_cores, stats_reduced_cb_config);

        // cb_var
        uint32_t cb_var_index = tt::CBIndex::c_19;
        tt::tt_metal::CircularBufferConfig cb_var_config =
            tt::tt_metal::CircularBufferConfig(ex_global_CB_size, {{cb_var_index, cb_data_format}})
                .set_page_size(cb_var_index, single_tile_size);
        tt::tt_metal::CreateCircularBuffer(program, sender_cores, cb_var_config);
    }

    // out
    uint32_t output_cb_index = tt::CBIndex::c_16;
    tt::tt_metal::CircularBufferConfig output_cb_config =
        tt::tt_metal::CircularBufferConfig(out_CB_size, {{output_cb_index, out_data_format}})
            .set_page_size(output_cb_index, out_single_tile_size);
    if (!is_post_all_gather || skip_write_back) {
        output_cb_config = output_cb_config.set_globally_allocated_address(*output.buffer());
    }
    CBHandle cb_output = 0;
    if (is_pre_all_gather) {
        cb_output = tt::tt_metal::CreateCircularBuffer(program, sender_cores, output_cb_config);
    } else {
        cb_output = tt::tt_metal::CreateCircularBuffer(program, all_cores, output_cb_config);
    }

    uint32_t output_reshard_cb_index = tt::CBIndex::c_17;
    tt::tt_metal::CircularBufferConfig output_reshard_cb_config =
        tt::tt_metal::CircularBufferConfig(out_reshard_CB_size, {{output_reshard_cb_index, out_data_format}})
            .set_page_size(output_reshard_cb_index, out_single_tile_size);
    if (is_post_all_gather && !skip_write_back) {
        output_reshard_cb_config = output_reshard_cb_config.set_globally_allocated_address(*output.buffer());

        tt::tt_metal::CreateCircularBuffer(program, all_worker_and_storage_cores, output_reshard_cb_config);
    }

    const auto& cores = corerange_to_cores(all_cores, all_cores.num_cores(), row_wise);

    // Runtime Args
    std::vector<KernelHandle> writer_kernel_ids;
    writer_kernel_ids.reserve(cores.size());
    float winv = 1.0f / block_w;
    float cinv = is_post_all_gather ? (1.0f / num_distributed_devices) : (1.0f / num_blocks);  // bcast-cores scaler
    float cinv_one = 1.0f;  // bcast-cores scaler for all-to-all cores not on first row/col
    auto bfloat_cinv_value = bfloat16(cinv);
    uint32_t packed_cinv_value = pack_two_bfloat16_into_uint32({bfloat_cinv_value, bfloat_cinv_value});
    auto bfloat_cinv_value_one = bfloat16(cinv_one);
    uint32_t packed_cinv_value_one = pack_two_bfloat16_into_uint32({bfloat_cinv_value_one, bfloat_cinv_value_one});
    auto bfloat_winv_value = bfloat16(winv);
    uint32_t packed_winv_value = pack_two_bfloat16_into_uint32({bfloat_winv_value, bfloat_winv_value});

    std::vector<uint32_t> in0_mcast_noc_x;
    std::vector<uint32_t> in0_mcast_noc_y;
    in0_mcast_noc_x.reserve(num_cores_x);
    in0_mcast_noc_y.reserve(num_cores_y);
    CoreCoord core_start_offset = grid_offset.value_or(CoreCoord{0, 0});
    for (uint32_t core_idx_x = core_start_offset.x; core_idx_x < num_cores_x + core_start_offset.x; ++core_idx_x) {
        in0_mcast_noc_x.push_back(device->worker_core_from_logical_core({core_idx_x, core_start_offset.y}).x);
    }
    for (uint32_t core_idx_y = core_start_offset.y; core_idx_y < num_cores_y + core_start_offset.y; ++core_idx_y) {
        in0_mcast_noc_y.push_back(device->worker_core_from_logical_core({core_start_offset.x, core_idx_y}).y);
    }

    uint32_t last_core_width_index = 0;
    if (!mcast_1d) {
        last_core_width_index = row_wise ? (num_cores_x - 1) : (num_cores_y - 1);
    } else {
        last_core_width_index = cores.size() - 1;
    }

    // For write back calculation
    uint32_t current_storage_core = 0;
    uint32_t current_storage_core_offset = 0;

    for (uint32_t i = 0; i < cores.size(); ++i) {
        const auto& core = cores[i];

        log_debug(tt::LogOp, "core: {}, {}", core.x, core.y);

        uint32_t height_index = 0, width_index = 0;
        if (mcast_1d) {
            height_index = 0;
            width_index = i;
        } else {
            if (row_wise) {
                height_index = core.y;
                width_index = core.x;
            } else {
                height_index = core.x;
                width_index = core.y;
            }
        }

        uint32_t width_index_two_stage = width_index % num_blocks_first_stage;

        uint32_t all_to_all_worker_tile_offset_size_bytes;
        if (use_two_stage_reduce) {
            all_to_all_worker_tile_offset_size_bytes =
                (width_index_two_stage * num_rows_per_all_to_all_worker) * single_tile_size;
        } else {
            all_to_all_worker_tile_offset_size_bytes =
                (width_index * num_rows_per_all_to_all_worker) * single_tile_size;
        }
        uint32_t gamma_tile_start_id = width_index * block_wt;
        uint32_t beta_tile_start_id = width_index * block_wt;

        uint32_t num_reduce_tiles_per_block_h = block_wt;
        // account for padding
        if (width_index == last_core_width_index) {
            num_reduce_tiles_per_block_h = Kt - last_core_width_index * block_wt;
        }

        std::vector<uint32_t> compute_args{num_reduce_tiles_per_block_h};
        if ((not use_two_stage_reduce and width_index < num_cores_all_to_all) or
            (use_two_stage_reduce and width_index_two_stage < num_cores_all_to_all_first_stage)) {
            uint32_t num_rows;
            if (use_two_stage_reduce) {
                num_rows = width_index_two_stage == num_cores_all_to_all_first_stage - 1
                               ? num_rows_per_all_to_all_worker_last
                               : num_rows_per_all_to_all_worker;
            } else {
                num_rows = width_index == num_cores_all_to_all - 1 ? num_rows_per_all_to_all_worker_last
                                                                   : num_rows_per_all_to_all_worker;
            }
            compute_args.push_back(num_rows);
            compute_args.push_back((uint32_t)use_two_stage_reduce);
            bool is_second_stage_reader;
            if (use_two_stage_reduce) {
                is_second_stage_reader = width_index < num_cores_all_to_all_first_stage;
            } else {
                is_second_stage_reader = false;
            }
            compute_args.push_back((uint32_t)is_second_stage_reader);
            if (is_post_all_gather) {
                compute_args.push_back((uint32_t)num_distributed_devices);
            }
            tt::tt_metal::SetRuntimeArgs(program, compute_kernels_id_all_to_all, core, compute_args);
        } else {
            tt::tt_metal::SetRuntimeArgs(program, compute_kernels_id, core, compute_args);
        }

        if (width_index == 0) {
            CoreCoord mcast_start, mcast_end;
            if (mcast_1d) {
                CoreCoord top_left_core = {(std::size_t)start_core.x, (std::size_t)start_core.y};
                CoreCoord bottom_right_core = {
                    (std::size_t)start_core.x + num_cores_x - 1, (std::size_t)start_core.y + num_cores_y - 1};
                auto top_left_core_physical = device->worker_core_from_logical_core(top_left_core);
                auto bottom_right_core_physical = device->worker_core_from_logical_core(bottom_right_core);
                mcast_start = top_left_core_physical;
                mcast_end = bottom_right_core_physical;
            } else {
                if (row_wise) {
                    CoreCoord left_core_plus_one = {(std::size_t)start_core.x + 1, (std::size_t)core.y};
                    CoreCoord right_core = {(std::size_t)start_core.x + num_cores_x - 1, (std::size_t)core.y};
                    auto left_core_plus_one_physical = device->worker_core_from_logical_core(left_core_plus_one);
                    auto right_core_physical = device->worker_core_from_logical_core(right_core);
                    mcast_start = left_core_plus_one_physical;
                    mcast_end = right_core_physical;
                } else {
                    CoreCoord top_core_plus_one = {(std::size_t)core.x, (std::size_t)start_core.y + 1};
                    CoreCoord bottom_core = {(std::size_t)core.x, (std::size_t)start_core.y + num_cores_y - 1};
                    auto top_core_plus_one_physical = device->worker_core_from_logical_core(top_core_plus_one);
                    auto bottom_core_physical = device->worker_core_from_logical_core(bottom_core);
                    mcast_start = top_core_plus_one_physical;
                    mcast_end = bottom_core_physical;
                }
            }
            if (reader_noc == NOC::NOC_1) {
                std::swap(mcast_start, mcast_end);
            }
            std::vector<uint32_t> mcast_sender_args;
            mcast_sender_args.push_back(mcast_start.x);
            mcast_sender_args.push_back(mcast_start.y);
            mcast_sender_args.push_back(mcast_end.x);
            mcast_sender_args.push_back(mcast_end.y);
            if (mcast_1d) {
                mcast_sender_args.push_back(core.x - start_core.x);
                mcast_sender_args.push_back(core.y - start_core.y);
                mcast_sender_args.insert(mcast_sender_args.end(), in0_mcast_noc_x.begin(), in0_mcast_noc_x.end());
                mcast_sender_args.insert(mcast_sender_args.end(), in0_mcast_noc_y.begin(), in0_mcast_noc_y.end());
            } else {
                if (row_wise) {
                    mcast_sender_args.push_back(core.x - start_core.x);
                    mcast_sender_args.push_back(0);
                    mcast_sender_args.insert(mcast_sender_args.end(), in0_mcast_noc_x.begin(), in0_mcast_noc_x.end());
                    mcast_sender_args.push_back(in0_mcast_noc_y[height_index]);
                } else {
                    mcast_sender_args.push_back(0);
                    mcast_sender_args.push_back(core.y - start_core.y);
                    mcast_sender_args.push_back(in0_mcast_noc_x[height_index]);
                    mcast_sender_args.insert(mcast_sender_args.end(), in0_mcast_noc_y.begin(), in0_mcast_noc_y.end());
                }
            }
            tt::tt_metal::SetRuntimeArgs(program, reader_mcast_sender_kernels_id, core, mcast_sender_args);
        } else if (
            (not use_two_stage_reduce and width_index < num_cores_all_to_all) or
            (use_two_stage_reduce and width_index_two_stage < num_cores_all_to_all_first_stage)) {
            std::vector<uint32_t> mcast_receiver_args;
            bool is_last_all_to_all_worker;
            if (use_two_stage_reduce) {
                is_last_all_to_all_worker = width_index_two_stage == num_cores_all_to_all_first_stage - 1;
            } else {
                is_last_all_to_all_worker = width_index == num_cores_all_to_all - 1;
            }
            mcast_receiver_args.push_back(is_last_all_to_all_worker);
            mcast_receiver_args.push_back(all_to_all_worker_tile_offset_size_bytes);
            bool is_second_stage_reader;
            if (use_two_stage_reduce and width_index < num_cores_all_to_all_first_stage) {
                is_second_stage_reader = true;
                mcast_receiver_args.push_back((uint32_t)is_second_stage_reader);
            } else {
                is_second_stage_reader = false;
                mcast_receiver_args.push_back((uint32_t)is_second_stage_reader);
            }
            if (mcast_1d) {
                mcast_receiver_args.push_back(core.x - start_core.x);
                mcast_receiver_args.push_back(core.y - start_core.y);
                mcast_receiver_args.insert(mcast_receiver_args.end(), in0_mcast_noc_x.begin(), in0_mcast_noc_x.end());
                mcast_receiver_args.insert(mcast_receiver_args.end(), in0_mcast_noc_y.begin(), in0_mcast_noc_y.end());
            } else {
                if (row_wise) {
                    mcast_receiver_args.push_back(core.x - start_core.x);
                    mcast_receiver_args.push_back(0);
                    mcast_receiver_args.insert(
                        mcast_receiver_args.end(), in0_mcast_noc_x.begin(), in0_mcast_noc_x.end());
                    mcast_receiver_args.push_back(in0_mcast_noc_y[height_index]);
                } else {
                    mcast_receiver_args.push_back(0);
                    mcast_receiver_args.push_back(core.y - start_core.y);
                    mcast_receiver_args.push_back(in0_mcast_noc_x[height_index]);
                    mcast_receiver_args.insert(
                        mcast_receiver_args.end(), in0_mcast_noc_y.begin(), in0_mcast_noc_y.end());
                }
            }
            tt::tt_metal::SetRuntimeArgs(
                program, reader_mcast_receiver_kernels_id_all_to_all, core, mcast_receiver_args);
        } else {
            std::vector<uint32_t> mcast_receiver_args;
            bool is_last_all_to_all_worker = false;
            mcast_receiver_args.push_back(is_last_all_to_all_worker);
            mcast_receiver_args.push_back(all_to_all_worker_tile_offset_size_bytes);
            mcast_receiver_args.push_back(0);
            mcast_receiver_args.push_back(0);
            mcast_receiver_args.push_back(0);
            if (mcast_1d) {
                mcast_receiver_args.push_back(in0_mcast_noc_x[0]);
                mcast_receiver_args.push_back(in0_mcast_noc_y[0]);
            } else {
                if (row_wise) {
                    mcast_receiver_args.push_back(in0_mcast_noc_x[0]);
                    mcast_receiver_args.push_back(in0_mcast_noc_y[height_index]);
                } else {
                    mcast_receiver_args.push_back(in0_mcast_noc_x[height_index]);
                    mcast_receiver_args.push_back(in0_mcast_noc_y[0]);
                }
            }
            tt::tt_metal::SetRuntimeArgs(program, reader_mcast_receiver_kernels_id, core, mcast_receiver_args);
        }

        // Compute write back rt args

        std::vector<uint32_t> write_back_writer_args;

        if (is_post_all_gather) {
            uint32_t num_storage_cores = all_storage_cores.num_cores();

            write_back_writer_args.push_back(
                current_storage_core_offset * out_single_tile_size);  // storage_core_start_offset

            uint32_t current_worker_num_segments_to_write_back = 0;
            uint32_t worker_core_current_offset = 0;

            while (worker_core_current_offset <
                   block_wt) {  // Continue until all worker core data has been written to corresponding storage cores
                uint32_t num_tiles_available_at_current_storage_core = block_wt_resharded - current_storage_core_offset;
                uint32_t num_tiles_left_on_current_worker_core = block_wt - worker_core_current_offset;
                uint32_t num_tiles_to_write_back =
                    std::min(num_tiles_left_on_current_worker_core, num_tiles_available_at_current_storage_core);
                current_worker_num_segments_to_write_back += 1;

                log_debug(
                    tt::LogOp,
                    "New segment for worker core {}, Worker core offset: {}, Storage core offset: {}, Num tiles to "
                    "write "
                    "back: {}",
                    i,
                    worker_core_current_offset,
                    current_storage_core_offset,
                    num_tiles_to_write_back);

                write_back_writer_args.push_back(
                    num_tiles_to_write_back * out_single_tile_size);  // num_bytes_to_write_back
                write_back_writer_args.push_back(
                    storage_core_noc_x[current_storage_core]);  // current_storage_core_noc_x
                write_back_writer_args.push_back(
                    storage_core_noc_y[current_storage_core]);  // current_storage_core_noc_y

                worker_core_current_offset += num_tiles_to_write_back;
                current_storage_core_offset += num_tiles_to_write_back;

                if (current_storage_core_offset >= block_wt_resharded) {
                    current_storage_core += 1;        // Move to next storage core
                    current_storage_core_offset = 0;  // Reset offset on new storage core

                    TT_FATAL(
                        current_storage_core <= num_storage_cores,
                        "current_storage_core {} is exceeding number of storage cores {}",
                        current_storage_core,
                        num_storage_cores);
                }
            }
            TT_FATAL(
                worker_core_current_offset == block_wt,
                "All worker core data should be written, but worker_core_current_offset {} != block_wt {}",
                worker_core_current_offset,
                block_wt);

            write_back_writer_args.insert(write_back_writer_args.begin(), current_worker_num_segments_to_write_back);
        }

        // Set writer runtime args
        if ((not use_two_stage_reduce and width_index < num_cores_all_to_all) or
            (use_two_stage_reduce and width_index_two_stage < num_cores_all_to_all_first_stage)) {
            std::vector<uint32_t> writer_mcast_sender_args;
            if (use_two_stage_reduce) {
                if (width_index < num_cores_all_to_all_first_stage) {
                    writer_mcast_sender_args.push_back(packed_cinv_value);
                    writer_mcast_sender_args.push_back(packed_winv_value);
                } else {
                    writer_mcast_sender_args.push_back(packed_cinv_value_one);
                    writer_mcast_sender_args.push_back(packed_winv_value);
                }
            } else {
                writer_mcast_sender_args.push_back(packed_cinv_value);
                writer_mcast_sender_args.push_back(packed_winv_value);
            }
            writer_mcast_sender_args.push_back(e.u);
            writer_mcast_sender_args.push_back(gamma_dram_addr);
            writer_mcast_sender_args.push_back(beta_dram_addr);
            writer_mcast_sender_args.push_back(gamma_tile_start_id);
            writer_mcast_sender_args.push_back(beta_tile_start_id);

            // Add args for write back (reshard)
            writer_mcast_sender_args.insert(
                writer_mcast_sender_args.end(), write_back_writer_args.begin(), write_back_writer_args.end());

            tt::tt_metal::SetRuntimeArgs(program, writer_mcast_sender_kernels_id, core, writer_mcast_sender_args);
            writer_kernel_ids.push_back(writer_mcast_sender_kernels_id);
        } else {
            std::vector<uint32_t> writer_mcast_receiver_args;
            writer_mcast_receiver_args.push_back(packed_cinv_value);
            writer_mcast_receiver_args.push_back(packed_winv_value);
            writer_mcast_receiver_args.push_back(e.u);
            writer_mcast_receiver_args.push_back(gamma_dram_addr);
            writer_mcast_receiver_args.push_back(beta_dram_addr);
            writer_mcast_receiver_args.push_back(gamma_tile_start_id);
            writer_mcast_receiver_args.push_back(beta_tile_start_id);

            // Add args for write back (reshard)
            writer_mcast_receiver_args.insert(
                writer_mcast_receiver_args.end(), write_back_writer_args.begin(), write_back_writer_args.end());

            tt::tt_metal::SetRuntimeArgs(program, writer_mcast_receiver_kernels_id, core, writer_mcast_receiver_args);
            writer_kernel_ids.push_back(writer_mcast_receiver_kernels_id);
        }
    }

    return cached_program_t{
        std::move(program),
        shared_variables_t{
            .writer_kernel_ids = writer_kernel_ids,
            .writer_mcast_sender_kernels_id = writer_mcast_sender_kernels_id,
            .writer_mcast_receiver_kernels_id = writer_mcast_receiver_kernels_id,
            .num_none_all_to_all_workers = num_none_all_to_all_workers,
            .is_pre_all_gather = is_pre_all_gather,
            .cb_in0 = cb_in0,
            .cb_in1 = cb_in1,
            .cb_stats = cb_stats,
            .cb_add_out = cb_add_out,
            .cb_output = cb_output,
            .cores = cores}};
}

void LayerNormShardedProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const LayerNormParams& /*operation_attributes*/,
    const LayerNormInputs& tensor_args,
    Tensor& tensor_return_value) {
    auto* const src_buffer_a = tensor_args.input.buffer();
    const auto& b_tensor = tensor_args.residual_input_tensor;
    const auto& gamma_tensor = tensor_args.weight;
    const auto& beta_tensor = tensor_args.bias;
    const auto& stats_tensor = tensor_args.stats;
    auto* const dst_buffer = tensor_return_value.buffer();

    const auto& capture = cached_program.shared_variables;
    auto& program = cached_program.program;

    UpdateDynamicCircularBufferAddress(program, capture.cb_in0, *src_buffer_a);

    if (b_tensor.has_value()) {
        UpdateDynamicCircularBufferAddress(program, capture.cb_in1, *b_tensor.value().buffer());
        if (capture.is_pre_all_gather) {
            UpdateDynamicCircularBufferAddress(program, capture.cb_add_out, *src_buffer_a);
        }
    }
    if (stats_tensor.has_value()) {
        UpdateDynamicCircularBufferAddress(program, capture.cb_stats, *stats_tensor.value().buffer());
    }

    UpdateDynamicCircularBufferAddress(program, capture.cb_output, *dst_buffer);

    auto& writer_sender_args_by_core = GetRuntimeArgs(program, capture.writer_mcast_sender_kernels_id);
    auto& writer_receiver_args_by_core = capture.num_none_all_to_all_workers > 0
                                             ? GetRuntimeArgs(program, capture.writer_mcast_receiver_kernels_id)
                                             : writer_sender_args_by_core;

    const auto gamma_address = gamma_tensor.has_value() ? gamma_tensor.value().buffer()->address() : 0;
    const auto beta_address = beta_tensor.has_value() ? beta_tensor.value().buffer()->address() : 0;

    for (uint32_t i = 0; i < capture.cores.size(); ++i) {
        const CoreCoord& core = capture.cores[i];

        const auto writer_kernel_id = capture.writer_kernel_ids.at(i);

        if (writer_kernel_id == capture.writer_mcast_sender_kernels_id) {
            auto& runtime_args = writer_sender_args_by_core[core.x][core.y];
            runtime_args[3] = gamma_address;
            runtime_args[4] = beta_address;

        } else if (writer_kernel_id == capture.writer_mcast_receiver_kernels_id) {
            auto& runtime_args = writer_receiver_args_by_core[core.x][core.y];
            runtime_args[3] = gamma_address;
            runtime_args[4] = beta_address;
        }
    }
}

}  // namespace ttnn::prim

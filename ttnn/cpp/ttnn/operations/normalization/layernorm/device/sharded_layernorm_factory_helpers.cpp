// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/normalization/layernorm/device/sharded_layernorm_factory_helpers.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim::sharded_layernorm_helpers {

//////////////////////////////////////////////////////////////////////////////
// Validation and data format helpers
//////////////////////////////////////////////////////////////////////////////

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

namespace {

// Internal helper: determines if two-stage reduce optimization should be used
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

// Internal helper: computes number of blocks based on grid configuration
uint32_t get_num_blocks(bool mcast_1d, bool row_wise, CoreCoord grid_size, const ShardSpec& shard_spec) {
    if (mcast_1d) {
        return shard_spec.num_cores();
    }
    if (row_wise) {
        return grid_size.x;
    }
    return grid_size.y;
}

}  // namespace

//////////////////////////////////////////////////////////////////////////////
// Grid and worker distribution
//////////////////////////////////////////////////////////////////////////////

GridParams GridParams::compute(const Tensor& input, uint32_t block_ht, CoreCoord compute_with_storage_grid_size) {
    auto spec = input.shard_spec().value();
    uint32_t M = input.physical_volume() / input.padded_shape()[-1];
    uint32_t block_h = block_ht * TILE_HEIGHT;
    bool mcast = M == block_h;
    bool rw = spec.orientation == ShardOrientation::ROW_MAJOR;
    auto bbox = spec.grid.bounding_box();
    CoreCoord gs = {bbox.end_coord.x - bbox.start_coord.x + 1, bbox.end_coord.y - bbox.start_coord.y + 1};
    std::optional<CoreCoord> offset = std::nullopt;
    if (bbox.start_coord.x != 0 || bbox.start_coord.y != 0) {
        offset = bbox.start_coord;
    }
    uint32_t nb = get_num_blocks(mcast, rw, gs, spec);
    return GridParams{
        .shard_spec = spec,
        .grid_size = gs,
        .grid_offset = offset,
        .mcast_1d = mcast,
        .row_wise = rw,
        .num_blocks = nb,
        .use_mcast = nb > 1,
        .use_two_stage_reduce = should_use_two_stage_reduce(mcast, rw, gs, compute_with_storage_grid_size)};
}

WorkerDistribution WorkerDistribution::compute(const GridParams& grid, uint32_t block_ht) {
    WorkerDistribution w;
    w.num_rows_per_all_to_all_worker = tt::div_up(block_ht, grid.num_blocks);
    if (grid.use_two_stage_reduce) {
        if (grid.row_wise) {
            w.num_rows_per_all_to_all_worker = tt::div_up(block_ht, grid.grid_size.x);
        } else {
            w.num_rows_per_all_to_all_worker = tt::div_up(block_ht, grid.grid_size.y);
        }
    }
    w.num_rows_per_all_to_all_worker_last =
        block_ht - ((block_ht / w.num_rows_per_all_to_all_worker) * w.num_rows_per_all_to_all_worker);

    w.num_cores_all_to_all = tt::div_up(block_ht, w.num_rows_per_all_to_all_worker);
    w.num_cores_all_to_all_first_stage = w.num_cores_all_to_all;
    w.num_cores_all_to_all_second_stage = 0;
    w.num_blocks_first_stage = grid.num_blocks;
    w.num_blocks_second_stage = 0;

    if (grid.use_two_stage_reduce) {
        if (grid.row_wise) {
            w.num_blocks_first_stage = grid.grid_size.x;
            w.num_cores_all_to_all_second_stage = grid.grid_size.y;
            w.num_cores_all_to_all *= grid.grid_size.y;
        } else {
            w.num_blocks_first_stage = grid.grid_size.y;
            w.num_cores_all_to_all_second_stage = grid.grid_size.x;
            w.num_cores_all_to_all *= grid.grid_size.x;
        }
        w.num_blocks_second_stage = w.num_cores_all_to_all_second_stage;
    }

    w.num_none_all_to_all_workers = grid.num_blocks - w.num_cores_all_to_all;
    if (w.num_rows_per_all_to_all_worker_last == 0) {
        w.num_rows_per_all_to_all_worker_last = w.num_rows_per_all_to_all_worker;
    }
    return w;
}

//////////////////////////////////////////////////////////////////////////////
// Core range computation
//////////////////////////////////////////////////////////////////////////////

CoreRangeSet apply_grid_offset(const CoreRangeSet& input_set, const CoreCoord& offset) {
    if (input_set.empty()) {
        return input_set;
    }
    std::vector<CoreRange> new_ranges;
    new_ranges.reserve(input_set.size());
    for (const CoreRange& range : input_set.ranges()) {
        CoreCoord new_start = {range.start_coord.x + offset.x, range.start_coord.y + offset.y};
        CoreCoord new_end = {range.end_coord.x + offset.x, range.end_coord.y + offset.y};
        new_ranges.emplace_back(new_start, new_end);
    }
    return CoreRangeSet(std::move(new_ranges));
}

CoreRanges compute_core_ranges_mcast_1d_row_wise(
    const GridParams& grid, const WorkerDistribution& workers, CoreCoord start_core) {
    CoreRanges cr;
    cr.start_core = start_core;
    cr.all_cores = grid.shard_spec.grid.merge_ranges();
    cr.sender_cores = {start_core, start_core};

    auto bbox = grid.shard_spec.grid.bounding_box();
    CoreCoord all_core_grid_size;
    CoreCoord none_core_grid_size;
    if (grid.use_two_stage_reduce) {
        all_core_grid_size = {workers.num_cores_all_to_all_first_stage, grid.grid_size.y};
        none_core_grid_size = {grid.grid_size.x - workers.num_cores_all_to_all_first_stage, grid.grid_size.y};
    } else {
        all_core_grid_size = grid.grid_size;
        none_core_grid_size = grid.grid_size;
    }

    cr.all_to_all_cores = num_cores_to_corerangeset(start_core, workers.num_cores_all_to_all, all_core_grid_size, true);

    if (grid.use_mcast) {
        CoreCoord all_start_core;
        CoreCoord end_core = cr.sender_cores.end_coord;
        if (grid.use_two_stage_reduce) {
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
        cr.all_to_all_workers_except_sender =
            num_cores_to_corerangeset(all_start_core, workers.num_cores_all_to_all - 1, all_core_grid_size, true);
    }

    if (workers.num_none_all_to_all_workers > 0) {
        if (grid.use_two_stage_reduce) {
            CoreCoord none_start_core = {all_core_grid_size.x, cr.sender_cores.end_coord.y};
            CoreCoord none_end_core = {grid.grid_size.x - 1, grid.grid_size.y - 1};
            cr.not_all_to_all_workers = CoreRangeSet(CoreRange(none_start_core, none_end_core));
        } else {
            CoreCoord none_start_core;
            CoreCoord end_core = (*cr.all_to_all_cores.ranges().rbegin()).end_coord;
            if (end_core.x == bbox.end_coord.x) {
                none_start_core = {0, end_core.y + 1};
            } else {
                none_start_core = {end_core.x + 1, end_core.y};
            }
            cr.not_all_to_all_workers = num_cores_to_corerangeset(
                none_start_core, workers.num_none_all_to_all_workers, none_core_grid_size, true);
        }
    }

    cr.num_cores_x_mcast = grid.grid_size.x;
    cr.num_cores_y_mcast = grid.grid_size.y;
    return cr;
}

CoreRanges compute_core_ranges_mcast_1d_col_wise(
    const GridParams& grid, const WorkerDistribution& workers, CoreCoord start_core) {
    CoreRanges cr;
    cr.start_core = start_core;
    cr.all_cores = grid.shard_spec.grid.merge_ranges();
    cr.sender_cores = {start_core, start_core};

    auto bbox = grid.shard_spec.grid.bounding_box();
    CoreCoord all_core_grid_size;
    CoreCoord none_core_grid_size;
    if (grid.use_two_stage_reduce) {
        all_core_grid_size = {grid.grid_size.x, workers.num_cores_all_to_all_first_stage};
        none_core_grid_size = {grid.grid_size.x, grid.grid_size.y - workers.num_cores_all_to_all_first_stage};
    } else {
        all_core_grid_size = grid.grid_size;
        none_core_grid_size = grid.grid_size;
    }

    cr.all_to_all_cores =
        num_cores_to_corerangeset(start_core, workers.num_cores_all_to_all, all_core_grid_size, false);

    if (grid.use_mcast) {
        CoreCoord all_start_core;
        CoreCoord end_core = cr.sender_cores.end_coord;
        if (grid.use_two_stage_reduce) {
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
        cr.all_to_all_workers_except_sender = num_cores_to_corerangeset(
            CoreCoord{start_core.x, start_core.y + 1}, workers.num_cores_all_to_all - 1, all_core_grid_size, false);
    }

    if (workers.num_none_all_to_all_workers > 0) {
        if (grid.use_two_stage_reduce) {
            CoreCoord none_start_core = {cr.sender_cores.end_coord.x, all_core_grid_size.y};
            CoreCoord none_end_core = {grid.grid_size.x - 1, grid.grid_size.y - 1};
            cr.not_all_to_all_workers = CoreRangeSet(CoreRange(none_start_core, none_end_core));
        } else {
            CoreCoord none_start_core;
            CoreCoord end_core = (*cr.all_to_all_cores.ranges().rbegin()).end_coord;
            if (end_core.y == bbox.end_coord.y) {
                none_start_core = {end_core.x + 1, 0};
            } else {
                none_start_core = {end_core.x, end_core.y + 1};
            }
            cr.not_all_to_all_workers = num_cores_to_corerangeset(
                none_start_core, workers.num_none_all_to_all_workers, none_core_grid_size, false);
        }
    }

    cr.num_cores_x_mcast = grid.grid_size.x;
    cr.num_cores_y_mcast = grid.grid_size.y;
    return cr;
}

CoreRanges compute_core_ranges_2d(const GridParams& grid, const WorkerDistribution& workers, CoreCoord start_core) {
    CoreRanges cr;
    cr.start_core = start_core;
    cr.all_cores = grid.shard_spec.grid.merge_ranges();

    uint32_t num_cores_x = grid.grid_size.x;
    uint32_t num_cores_y = grid.grid_size.y;

    if (grid.row_wise) {
        cr.sender_cores = {
            {(std::size_t)start_core.x, (std::size_t)start_core.y},
            {(std::size_t)start_core.x, (std::size_t)start_core.y + num_cores_y - 1}};
        cr.all_to_all_cores = CoreRangeSet(CoreRange(
            {(std::size_t)start_core.x, (std::size_t)start_core.y},
            {(std::size_t)start_core.x + workers.num_cores_all_to_all - 1,
             (std::size_t)start_core.y + num_cores_y - 1}));
        if (grid.use_mcast && workers.num_cores_all_to_all > 1) {
            cr.all_to_all_workers_except_sender = CoreRangeSet(CoreRange(
                {(std::size_t)start_core.x + 1, (std::size_t)start_core.y},
                {(std::size_t)start_core.x + workers.num_cores_all_to_all - 1,
                 (std::size_t)start_core.y + num_cores_y - 1}));
        }
        if (workers.num_none_all_to_all_workers > 0) {
            cr.not_all_to_all_workers = CoreRangeSet(CoreRange(
                {(std::size_t)start_core.x + workers.num_cores_all_to_all, (std::size_t)start_core.y},
                {(std::size_t)start_core.x + num_cores_x - 1, (std::size_t)start_core.y + num_cores_y - 1}));
        }
        cr.num_cores_x_mcast = num_cores_x;
        cr.num_cores_y_mcast = 1;
    } else {
        cr.sender_cores = {
            {(std::size_t)start_core.x, (std::size_t)start_core.y},
            {(std::size_t)start_core.x + num_cores_x - 1, (std::size_t)start_core.y}};
        cr.all_to_all_cores = CoreRangeSet(CoreRange(
            {(std::size_t)start_core.x, (std::size_t)start_core.y},
            {(std::size_t)start_core.x + num_cores_x - 1,
             (std::size_t)start_core.y + workers.num_cores_all_to_all - 1}));
        if (grid.use_mcast && workers.num_cores_all_to_all > 1) {
            cr.all_to_all_workers_except_sender = CoreRangeSet(CoreRange(
                {(std::size_t)start_core.x, (std::size_t)start_core.y + 1},
                {(std::size_t)start_core.x + num_cores_x - 1,
                 (std::size_t)start_core.y + workers.num_cores_all_to_all - 1}));
        }
        if (workers.num_none_all_to_all_workers > 0) {
            cr.not_all_to_all_workers = CoreRangeSet(CoreRange(
                {(std::size_t)start_core.x, (std::size_t)start_core.y + workers.num_cores_all_to_all},
                {(std::size_t)start_core.x + num_cores_x - 1, (std::size_t)start_core.y + num_cores_y - 1}));
        }
        cr.num_cores_x_mcast = 1;
        cr.num_cores_y_mcast = num_cores_y;
    }
    return cr;
}

CoreRanges CoreRanges::compute(const GridParams& grid, const WorkerDistribution& workers) {
    CoreCoord start_core = {0, 0};
    CoreRanges cr;

    if (grid.mcast_1d) {
        if (grid.row_wise) {
            cr = compute_core_ranges_mcast_1d_row_wise(grid, workers, start_core);
        } else {
            cr = compute_core_ranges_mcast_1d_col_wise(grid, workers, start_core);
        }
    } else {
        cr = compute_core_ranges_2d(grid, workers, start_core);
    }

    // Apply grid offset if needed
    // Note: all_cores comes from grid.shard_spec.grid which already has the offset embedded,
    // so we don't apply the offset to it. Other ranges are computed from (0,0)-based
    // coordinates and need the offset applied.
    if (grid.grid_offset.has_value()) {
        const auto& offset = grid.grid_offset.value();
        cr.start_core = {cr.start_core.x + offset.x, cr.start_core.y + offset.y};
        cr.sender_cores = {
            {cr.sender_cores.start_coord.x + offset.x, cr.sender_cores.start_coord.y + offset.y},
            {cr.sender_cores.end_coord.x + offset.x, cr.sender_cores.end_coord.y + offset.y}};
        // Don't apply offset to all_cores - it comes from shard_spec.grid which already has the offset
        cr.all_to_all_cores = apply_grid_offset(cr.all_to_all_cores, offset);
        cr.all_to_all_workers_except_sender = apply_grid_offset(cr.all_to_all_workers_except_sender, offset);
        cr.not_all_to_all_workers = apply_grid_offset(cr.not_all_to_all_workers, offset);
    }

    return cr;
}

//////////////////////////////////////////////////////////////////////////////
// Kernel paths, defines, and compile-time args helpers
//////////////////////////////////////////////////////////////////////////////

KernelPaths KernelPaths::get(
    bool is_pre_all_gather, bool is_post_all_gather, bool use_row_major_kernel, bool use_welford) {
    KernelPaths paths;

    constexpr const char* base_path = "ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/";

    if (is_pre_all_gather) {
        paths.reader_sender =
            std::string(base_path) + "dataflow/reader_mcast_sender_unary_sharded_ln_pre_allgather.cpp";
        paths.reader_receiver =
            std::string(base_path) + "dataflow/reader_mcast_receiver_unary_sharded_ln_pre_allgather.cpp";
        paths.writer = std::string(base_path) + "dataflow/writer_unary_sharded_ln_pre_all_gather.cpp";
        paths.compute = std::string(base_path) + "compute/layernorm_sharded_pre_allgather.cpp";
    } else if (is_post_all_gather) {
        paths.reader_sender =
            std::string(base_path) + "dataflow/reader_mcast_sender_unary_sharded_ln_post_allgather.cpp";
        paths.reader_receiver =
            std::string(base_path) + "dataflow/reader_mcast_receiver_unary_sharded_ln_post_allgather.cpp";
        paths.writer = use_row_major_kernel ? std::string(base_path) + "dataflow/writer_unary_sharded_ln_rm_gb.cpp"
                                            : std::string(base_path) + "dataflow/writer_unary_sharded_ln.cpp";
        paths.compute = std::string(base_path) + "compute/layernorm_sharded_post_allgather.cpp";
    } else {
        paths.reader_sender = std::string(base_path) + "dataflow/reader_mcast_sender_unary_sharded_ln.cpp";
        paths.reader_receiver = std::string(base_path) + "dataflow/reader_mcast_receiver_unary_sharded_ln.cpp";
        paths.writer = use_row_major_kernel ? std::string(base_path) + "dataflow/writer_unary_sharded_ln_rm_gb.cpp"
                                            : std::string(base_path) + "dataflow/writer_unary_sharded_ln.cpp";
        paths.compute = use_welford ? std::string(base_path) + "compute/layernorm_sharded_welford.cpp"
                                    : std::string(base_path) + "compute/layernorm_sharded.cpp";
    }

    return paths;
}

KernelDefines KernelDefines::build(
    bool has_b, bool has_gamma, bool has_beta, bool rms_norm, bool use_welford, bool skip_write_back) {
    KernelDefines defines;

    // Reader defines
    if (has_b) {
        defines.reader.emplace_back("FUSE_PRE_ADD", "1");
    }
    if (has_gamma) {
        defines.reader.emplace_back("FUSE_GAMMA", "1");
    }
    if (has_beta) {
        defines.reader.emplace_back("FUSE_BETA", "1");
    }

    // Writer defines
    if (rms_norm) {
        defines.writer.emplace_back("RMSNORM", "1");
    }
    if (skip_write_back) {
        defines.writer.emplace_back("SKIP_WRITE_BACK", "1");
    }

    // Compute defines
    if (has_b) {
        defines.compute.emplace_back("FUSE_PRE_ADD", "1");
    }
    if (rms_norm && !use_welford) {
        defines.compute.emplace_back("RMSNORM", "1");
    }

    return defines;
}

CBSizeParams::Sizes CBSizeParams::compute() const {
    Sizes sizes;

    uint32_t in0_block_tiles = block_wt * block_ht;

    sizes.in0_CB_size = in0_block_tiles * in_single_tile_size;
    sizes.in1_CB_size = sizes.in0_CB_size;
    sizes.in2_CB_size = bfloat16_tile_size;
    sizes.in3_CB_size = bfloat16_tile_size;
    sizes.in5_CB_size = in0_block_tiles * gamma_single_tile_size / block_ht;
    sizes.in6_CB_size = in0_block_tiles * beta_single_tile_size / block_ht;

    sizes.x_CB_size = in0_block_tiles * single_tile_size;
    sizes.xmm_CB_size = in0_block_tiles * single_tile_size;

    sizes.ex_partial_CB_size = in0_block_tiles * single_tile_size / block_wt;
    sizes.ex_external_CB_size = tt::div_up(Kt, block_wt) * single_tile_size;

    if (is_pre_all_gather || is_post_all_gather) {
        sizes.ex_partial_CB_size = sizes.ex_partial_CB_size * pre_all_gather_stats_block_tiles;
    }

    sizes.ex_CB_size = sizes.ex_partial_CB_size;
    sizes.ex_global_CB_size = sizes.ex_partial_CB_size;
    sizes.ex2pe_CB_size = num_rows_per_all_to_all_worker * single_tile_size;

    if (is_post_all_gather) {
        sizes.stats_cb_size = post_all_gather_stats_block_tiles * single_tile_size;
        sizes.stats_reduced_cb_size = pre_all_gather_stats_block_tiles * single_tile_size;
    }

    if (is_pre_all_gather) {
        sizes.out_CB_size = pre_all_gather_stats_block_tiles * out_single_tile_size;
    } else {
        sizes.out_CB_size = in0_block_tiles * out_single_tile_size;
    }

    sizes.out_reshard_CB_size = sizes.out_CB_size;
    if (is_post_all_gather && !skip_write_back) {
        sizes.out_reshard_CB_size = block_wt_resharded * block_ht * out_single_tile_size;
    }

    // Update ex_external_CB_size based on configuration
    if (use_two_stage_reduce) {
        sizes.ex_external_CB_size = (num_blocks_first_stage + num_blocks_second_stage - 1) * single_tile_size;
    }
    if (is_pre_all_gather) {
        sizes.ex_external_CB_size = sizes.ex_external_CB_size * pre_all_gather_stats_block_tiles;
    }

    if (use_welford) {
        sizes.ex_external_CB_size *= 2;
        sizes.ex_partial_CB_size *= 2;
        sizes.ex_CB_size *= 2;
        sizes.ex_global_CB_size *= 2;
    }

    return sizes;
}

CompileTimeArgs CompileTimeArgs::build(const CompileTimeArgsContext& ctx) {
    CompileTimeArgs args;

    const auto& grid = *ctx.grid;
    const auto& workers = *ctx.workers;
    const auto& core_ranges = *ctx.core_ranges;

    uint32_t num_subblocks_w = ctx.block_wt / ctx.subblock_wt;

    // Reader sender compile time args
    args.reader_sender = {
        ctx.reduce_receiver_semaphore_id,
        ctx.reduce_sender_semaphore_id,
        grid.num_blocks,
        ctx.block_ht,
        ctx.block_ht * ctx.single_tile_size,
        workers.num_cores_all_to_all_first_stage,
        workers.num_rows_per_all_to_all_worker,
        workers.num_rows_per_all_to_all_worker * ctx.single_tile_size,
        workers.num_rows_per_all_to_all_worker_last,
        workers.num_rows_per_all_to_all_worker_last * ctx.single_tile_size,
        (uint32_t)grid.row_wise,
        core_ranges.num_cores_x_mcast,
        core_ranges.num_cores_y_mcast,
        (uint32_t)grid.use_two_stage_reduce,
        workers.num_blocks_first_stage,
        workers.num_blocks_second_stage,
        ctx.reduce_second_stage_semaphore_id,
        (uint32_t)ctx.rms_norm,
        (uint32_t)ctx.use_welford};

    // Reader receiver all-to-all compile time args
    args.reader_receiver_all_to_all = {
        ctx.reduce_receiver_semaphore_id,
        ctx.reduce_sender_semaphore_id,
        grid.num_blocks,
        ctx.block_ht,
        1,  // is_all_to_all_worker
        workers.num_cores_all_to_all_first_stage,
        workers.num_rows_per_all_to_all_worker,
        workers.num_rows_per_all_to_all_worker_last,
        (uint32_t)grid.row_wise,
        core_ranges.num_cores_x_mcast,
        core_ranges.num_cores_y_mcast,
        (uint32_t)grid.use_two_stage_reduce,
        workers.num_blocks_first_stage,
        workers.num_blocks_second_stage,
        ctx.reduce_second_stage_semaphore_id,
        (uint32_t)ctx.rms_norm,
        (uint32_t)ctx.use_welford};

    // Reader receiver (not all-to-all) compile time args
    args.reader_receiver = {
        ctx.reduce_receiver_semaphore_id,
        ctx.reduce_sender_semaphore_id,
        grid.num_blocks,
        ctx.block_ht,
        0,  // is_all_to_all_worker
        workers.num_cores_all_to_all_first_stage,
        workers.num_rows_per_all_to_all_worker,
        workers.num_rows_per_all_to_all_worker_last,
        (uint32_t)grid.row_wise,
        1,  // num_cores_x_mcast (dummy for non-all-to-all)
        1,  // num_cores_y_mcast (dummy for non-all-to-all)
        0,  // use_two_stage_reduce
        0,  // num_blocks_first_stage
        0,  // num_blocks_second_stage
        ctx.reduce_second_stage_semaphore_id,
        (uint32_t)ctx.rms_norm,
        (uint32_t)ctx.use_welford};

    // Writer sender compile time args
    args.writer_sender = {
        1,  // is_all_to_all_worker
        (uint32_t)ctx.has_gamma,
        (uint32_t)ctx.has_beta,
        ctx.block_wt,
        (uint32_t)ctx.use_welford};
    tt::tt_metal::TensorAccessorArgs(ctx.gamma_buffer).append_to(args.writer_sender);
    tt::tt_metal::TensorAccessorArgs(ctx.beta_buffer).append_to(args.writer_sender);

    // Writer receiver compile time args
    args.writer_receiver = {
        0,  // is_all_to_all_worker
        (uint32_t)ctx.has_gamma,
        (uint32_t)ctx.has_beta,
        ctx.block_wt,
        (uint32_t)ctx.use_welford};
    tt::tt_metal::TensorAccessorArgs(ctx.gamma_buffer).append_to(args.writer_receiver);
    tt::tt_metal::TensorAccessorArgs(ctx.beta_buffer).append_to(args.writer_receiver);

    // Add stick size for row-major gamma/beta
    if (ctx.gamma_is_row_major) {
        args.writer_sender.push_back(ctx.gamma_stick_size);
        args.writer_receiver.push_back(ctx.gamma_stick_size);
    } else if (ctx.beta_is_row_major) {
        args.writer_sender.push_back(ctx.beta_stick_size);
        args.writer_receiver.push_back(ctx.beta_stick_size);
    }

    // Add data format flags
    args.writer_sender.push_back(ctx.gamma_cb_data_format == tt::DataFormat::Float32);
    args.writer_sender.push_back(ctx.beta_cb_data_format == tt::DataFormat::Float32);
    args.writer_receiver.push_back(ctx.gamma_cb_data_format == tt::DataFormat::Float32);
    args.writer_receiver.push_back(ctx.beta_cb_data_format == tt::DataFormat::Float32);

    // Write-back compile time args
    args.writer_sender.push_back(ctx.block_wt * ctx.out_single_tile_size);
    args.writer_sender.push_back(ctx.block_wt_resharded * ctx.out_single_tile_size);
    args.writer_sender.push_back(ctx.block_ht);

    args.writer_receiver.push_back(ctx.block_wt * ctx.out_single_tile_size);
    args.writer_receiver.push_back(ctx.block_wt_resharded * ctx.out_single_tile_size);
    args.writer_receiver.push_back(ctx.block_ht);
    args.writer_receiver.push_back(ctx.use_welford);

    // Compute compile time args (all-to-all)
    bool float32_reduction = ctx.fp32_dest_acc_en && !ctx.legacy_reduction;
    args.compute_all_to_all = {
        0,
        (uint32_t)ctx.has_gamma,
        (uint32_t)ctx.has_beta,
        workers.num_blocks_first_stage,
        ctx.block_ht,
        ctx.block_wt,
        ctx.subblock_wt,
        num_subblocks_w,
        1,  // is_all_to_all_worker
        ctx.block_ht * ctx.block_wt,
        (uint32_t)ctx.fp32_dest_acc_en,
        (uint32_t)float32_reduction,
        (uint32_t)ctx.legacy_rsqrt,
        workers.num_blocks_second_stage};

    // Compute compile time args (not all-to-all)
    args.compute_not_all_to_all = {
        0,
        (uint32_t)ctx.has_gamma,
        (uint32_t)ctx.has_beta,
        workers.num_blocks_first_stage,
        ctx.block_ht,
        ctx.block_wt,
        ctx.subblock_wt,
        num_subblocks_w,
        0,  // is_all_to_all_worker
        ctx.block_ht * ctx.block_wt,
        (uint32_t)ctx.fp32_dest_acc_en,
        (uint32_t)float32_reduction,
        (uint32_t)ctx.legacy_rsqrt,
        workers.num_blocks_second_stage};

    // Welford-specific compute args
    if (ctx.use_welford) {
        constexpr uint32_t tile_width = 32;  // TILE_WIDTH
        uint32_t last_tile_W = ctx.K - ((ctx.K - tile_width) / tile_width) * tile_width;
        auto eps_u32 = std::bit_cast<uint32_t>(ctx.eps);

        args.compute_all_to_all.push_back(tile_width);
        args.compute_all_to_all.push_back(last_tile_W);
        args.compute_all_to_all.push_back(ctx.K);
        args.compute_all_to_all.push_back(eps_u32);
        args.compute_all_to_all.push_back(ctx.per_core_recip_lut_size);

        args.compute_not_all_to_all.push_back(tile_width);
        args.compute_not_all_to_all.push_back(last_tile_W);
        args.compute_not_all_to_all.push_back(ctx.K);
        args.compute_not_all_to_all.push_back(eps_u32);
        args.compute_not_all_to_all.push_back(ctx.per_core_recip_lut_size);
    }

    return args;
}

//////////////////////////////////////////////////////////////////////////////
// Kernel and CB descriptor builders
//////////////////////////////////////////////////////////////////////////////

void add_kernel_descriptors(
    ProgramDescriptor& program_descriptor,
    const CoreRanges& core_ranges,
    const WorkerDistribution& workers,
    const GridParams& grid,
    KernelConfig&& kernel_config) {
    // Reader sender kernel
    KernelDescriptor reader_sender_kernel_desc;
    reader_sender_kernel_desc.kernel_source = kernel_config.reader_sender_path;
    reader_sender_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_sender_kernel_desc.core_ranges = core_ranges.sender_cores;
    reader_sender_kernel_desc.compile_time_args = std::move(kernel_config.reader_sender_ct_args);
    reader_sender_kernel_desc.defines = std::move(kernel_config.reader_sender_defines);
    reader_sender_kernel_desc.runtime_args = std::move(kernel_config.reader_sender_rt_args);
    reader_sender_kernel_desc.config = DataMovementConfigDescriptor{
        .processor = DataMovementProcessor::RISCV_0,
        .noc = kernel_config.reader_noc,
        .noc_mode = NOC_MODE::DM_DEDICATED_NOC};
    program_descriptor.kernels.push_back(std::move(reader_sender_kernel_desc));

    // Reader receiver all-to-all kernel
    if (grid.use_mcast && !core_ranges.all_to_all_workers_except_sender.empty()) {
        KernelDescriptor reader_receiver_all_to_all_kernel_desc;
        reader_receiver_all_to_all_kernel_desc.kernel_source = kernel_config.reader_receiver_path;
        reader_receiver_all_to_all_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
        reader_receiver_all_to_all_kernel_desc.core_ranges = core_ranges.all_to_all_workers_except_sender;
        reader_receiver_all_to_all_kernel_desc.compile_time_args =
            std::move(kernel_config.reader_receiver_all_to_all_ct_args);
        reader_receiver_all_to_all_kernel_desc.defines = kernel_config.reader_receiver_defines;
        reader_receiver_all_to_all_kernel_desc.runtime_args =
            std::move(kernel_config.reader_receiver_all_to_all_rt_args);
        reader_receiver_all_to_all_kernel_desc.config = DataMovementConfigDescriptor{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = kernel_config.reader_noc,
            .noc_mode = NOC_MODE::DM_DEDICATED_NOC};
        program_descriptor.kernels.push_back(std::move(reader_receiver_all_to_all_kernel_desc));
    }

    // Reader receiver (not all-to-all) kernel
    if (workers.num_none_all_to_all_workers > 0) {
        KernelDescriptor reader_receiver_kernel_desc;
        reader_receiver_kernel_desc.kernel_source = kernel_config.reader_receiver_path;
        reader_receiver_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
        reader_receiver_kernel_desc.core_ranges = core_ranges.not_all_to_all_workers;
        reader_receiver_kernel_desc.compile_time_args = std::move(kernel_config.reader_receiver_ct_args);
        reader_receiver_kernel_desc.defines = std::move(kernel_config.reader_receiver_defines);
        reader_receiver_kernel_desc.runtime_args = std::move(kernel_config.reader_receiver_rt_args);
        reader_receiver_kernel_desc.config = DataMovementConfigDescriptor{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = kernel_config.reader_noc,
            .noc_mode = NOC_MODE::DM_DEDICATED_NOC};
        program_descriptor.kernels.push_back(std::move(reader_receiver_kernel_desc));
    }

    // Writer sender kernel (for all-to-all cores)
    KernelDescriptor writer_sender_kernel_desc;
    writer_sender_kernel_desc.kernel_source = kernel_config.writer_path;
    writer_sender_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_sender_kernel_desc.core_ranges = core_ranges.all_to_all_cores;
    writer_sender_kernel_desc.compile_time_args = std::move(kernel_config.writer_sender_ct_args);
    writer_sender_kernel_desc.defines = kernel_config.writer_defines;
    writer_sender_kernel_desc.runtime_args = std::move(kernel_config.writer_sender_rt_args);
    writer_sender_kernel_desc.config = DataMovementConfigDescriptor{
        .processor = DataMovementProcessor::RISCV_1,
        .noc = kernel_config.writer_noc,
        .noc_mode = NOC_MODE::DM_DEDICATED_NOC};
    program_descriptor.kernels.push_back(std::move(writer_sender_kernel_desc));

    // Writer receiver kernel (for not all-to-all cores)
    if (workers.num_none_all_to_all_workers > 0) {
        KernelDescriptor writer_receiver_kernel_desc;
        writer_receiver_kernel_desc.kernel_source = kernel_config.writer_path;
        writer_receiver_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
        writer_receiver_kernel_desc.core_ranges = core_ranges.not_all_to_all_workers;
        writer_receiver_kernel_desc.compile_time_args = std::move(kernel_config.writer_receiver_ct_args);
        writer_receiver_kernel_desc.defines = std::move(kernel_config.writer_defines);
        writer_receiver_kernel_desc.runtime_args = std::move(kernel_config.writer_receiver_rt_args);
        writer_receiver_kernel_desc.config = DataMovementConfigDescriptor{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = kernel_config.writer_noc,
            .noc_mode = NOC_MODE::DM_DEDICATED_NOC};
        program_descriptor.kernels.push_back(std::move(writer_receiver_kernel_desc));
    }

    // Compute kernel (all-to-all cores)
    KernelDescriptor compute_all_to_all_kernel_desc;
    compute_all_to_all_kernel_desc.kernel_source = kernel_config.compute_path;
    compute_all_to_all_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_all_to_all_kernel_desc.core_ranges = core_ranges.all_to_all_cores;
    compute_all_to_all_kernel_desc.compile_time_args = std::move(kernel_config.compute_all_to_all_ct_args);
    compute_all_to_all_kernel_desc.defines = kernel_config.compute_defines;
    compute_all_to_all_kernel_desc.runtime_args = std::move(kernel_config.compute_all_to_all_rt_args);
    compute_all_to_all_kernel_desc.config = ComputeConfigDescriptor{
        .math_fidelity = kernel_config.math_fidelity,
        .fp32_dest_acc_en = kernel_config.fp32_dest_acc_en,
        .math_approx_mode = kernel_config.math_approx_mode};
    program_descriptor.kernels.push_back(std::move(compute_all_to_all_kernel_desc));

    // Compute kernel (not all-to-all cores)
    if (workers.num_none_all_to_all_workers > 0) {
        KernelDescriptor compute_not_all_to_all_kernel_desc;
        compute_not_all_to_all_kernel_desc.kernel_source = kernel_config.compute_path;
        compute_not_all_to_all_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
        compute_not_all_to_all_kernel_desc.core_ranges = core_ranges.not_all_to_all_workers;
        compute_not_all_to_all_kernel_desc.compile_time_args = std::move(kernel_config.compute_not_all_to_all_ct_args);
        compute_not_all_to_all_kernel_desc.defines = std::move(kernel_config.compute_defines);
        compute_not_all_to_all_kernel_desc.runtime_args = std::move(kernel_config.compute_not_all_to_all_rt_args);
        compute_not_all_to_all_kernel_desc.config = ComputeConfigDescriptor{
            .math_fidelity = kernel_config.math_fidelity,
            .fp32_dest_acc_en = kernel_config.fp32_dest_acc_en,
            .math_approx_mode = kernel_config.math_approx_mode};
        program_descriptor.kernels.push_back(std::move(compute_not_all_to_all_kernel_desc));
    }
}

void add_cb_descriptors(
    ProgramDescriptor& program_descriptor,
    const CoreRanges& core_ranges,
    const CoreRangeSet& all_worker_and_storage_cores,
    const CBConfig& cb_config) {
    auto make_cb_descriptor = [](uint32_t total_size,
                                 const CoreRangeSet& core_ranges,
                                 uint8_t buffer_index,
                                 tt::DataFormat data_format,
                                 uint32_t page_size,
                                 Buffer* buffer = nullptr) {
        CBDescriptor cb_desc;
        cb_desc.total_size = total_size;
        cb_desc.core_ranges = core_ranges;
        cb_desc.format_descriptors.push_back(
            CBFormatDescriptor{.buffer_index = buffer_index, .data_format = data_format, .page_size = page_size});
        cb_desc.buffer = buffer;
        return cb_desc;
    };

    // CB 0: in0 sharded
    program_descriptor.cbs.push_back(make_cb_descriptor(
        cb_config.in0_CB_size,
        core_ranges.all_cores,
        tt::CBIndex::c_0,
        cb_config.in_data_format,
        cb_config.in_single_tile_size,
        cb_config.a_buffer));

    // CB 1: in1 sharded (if b)
    if (cb_config.has_b) {
        program_descriptor.cbs.push_back(make_cb_descriptor(
            cb_config.in1_CB_size,
            core_ranges.all_cores,
            tt::CBIndex::c_1,
            cb_config.in_data_format,
            cb_config.in_single_tile_size,
            cb_config.b_buffer));
        if (cb_config.is_pre_all_gather) {
            program_descriptor.cbs.push_back(make_cb_descriptor(
                cb_config.in1_CB_size,
                core_ranges.all_cores,
                tt::CBIndex::c_14,
                cb_config.in_data_format,
                cb_config.in_single_tile_size,
                cb_config.a_buffer));
        }
    }

    // CB 5: gamma
    if (cb_config.has_gamma) {
        program_descriptor.cbs.push_back(make_cb_descriptor(
            cb_config.in5_CB_size,
            core_ranges.all_cores,
            tt::CBIndex::c_5,
            cb_config.gamma_cb_data_format,
            cb_config.gamma_single_tile_size));
    }

    // CB 6: beta
    if (cb_config.has_beta) {
        program_descriptor.cbs.push_back(make_cb_descriptor(
            cb_config.in6_CB_size,
            core_ranges.all_cores,
            tt::CBIndex::c_6,
            cb_config.beta_cb_data_format,
            cb_config.beta_single_tile_size));
    }

    // CB 24: x
    program_descriptor.cbs.push_back(make_cb_descriptor(
        cb_config.x_CB_size,
        core_ranges.all_cores,
        tt::CBIndex::c_24,
        cb_config.cb_data_format,
        cb_config.single_tile_size));

    // CB 18: xmm
    program_descriptor.cbs.push_back(make_cb_descriptor(
        cb_config.xmm_CB_size,
        core_ranges.all_cores,
        tt::CBIndex::c_18,
        cb_config.cb_data_format,
        cb_config.single_tile_size));

    // ex_partial, ex, ex_external (if not rms_norm)
    if (!cb_config.rms_norm) {
        program_descriptor.cbs.push_back(make_cb_descriptor(
            cb_config.ex_partial_CB_size,
            core_ranges.all_cores,
            tt::CBIndex::c_8,
            cb_config.cb_data_format,
            cb_config.single_tile_size));
        program_descriptor.cbs.push_back(make_cb_descriptor(
            cb_config.ex_CB_size,
            core_ranges.all_cores,
            tt::CBIndex::c_9,
            cb_config.cb_data_format,
            cb_config.single_tile_size));
        program_descriptor.cbs.push_back(make_cb_descriptor(
            cb_config.ex_external_CB_size,
            core_ranges.all_cores,
            tt::CBIndex::c_10,
            cb_config.cb_data_format,
            cb_config.single_tile_size));
    }

    if (!cb_config.use_welford) {
        // CB 2: in2 scaler
        program_descriptor.cbs.push_back(make_cb_descriptor(
            cb_config.in2_CB_size,
            core_ranges.all_cores,
            tt::CBIndex::c_2,
            tt::DataFormat::Float16_b,
            cb_config.bfloat16_tile_size));
        // CB 3: in3 eps
        program_descriptor.cbs.push_back(make_cb_descriptor(
            cb_config.in3_CB_size,
            core_ranges.all_cores,
            tt::CBIndex::c_3,
            tt::DataFormat::Float16_b,
            cb_config.bfloat16_tile_size));
        // CB 4: in4 scaler-c
        program_descriptor.cbs.push_back(make_cb_descriptor(
            cb_config.in2_CB_size,
            core_ranges.all_cores,
            tt::CBIndex::c_4,
            tt::DataFormat::Float16_b,
            cb_config.bfloat16_tile_size));
        // CB 11: ex_partial2
        program_descriptor.cbs.push_back(make_cb_descriptor(
            cb_config.ex_partial_CB_size,
            core_ranges.all_cores,
            tt::CBIndex::c_11,
            cb_config.cb_data_format,
            cb_config.single_tile_size));
        // CB 12: ex2
        program_descriptor.cbs.push_back(make_cb_descriptor(
            cb_config.ex_CB_size,
            core_ranges.all_cores,
            tt::CBIndex::c_12,
            cb_config.cb_data_format,
            cb_config.single_tile_size));
        // CB 13: ex_external2
        program_descriptor.cbs.push_back(make_cb_descriptor(
            cb_config.ex_external_CB_size,
            core_ranges.all_cores,
            tt::CBIndex::c_13,
            cb_config.cb_data_format,
            cb_config.single_tile_size));
        // CB 20: ex2pe
        program_descriptor.cbs.push_back(make_cb_descriptor(
            cb_config.ex2pe_CB_size,
            core_ranges.all_cores,
            tt::CBIndex::c_20,
            cb_config.cb_data_format,
            cb_config.single_tile_size));
    }

    // CB 15: ex_global
    program_descriptor.cbs.push_back(make_cb_descriptor(
        cb_config.ex_global_CB_size,
        core_ranges.all_cores,
        tt::CBIndex::c_15,
        cb_config.cb_data_format,
        cb_config.single_tile_size));

    if (cb_config.use_welford) {
        // CB 22: transpose intermediate
        program_descriptor.cbs.push_back(make_cb_descriptor(
            cb_config.ex_global_CB_size,
            core_ranges.all_cores,
            tt::CBIndex::c_22,
            cb_config.cb_data_format,
            cb_config.single_tile_size));

        // CB 25: Reciprocal LUT
        CBDescriptor recip_cb_desc;
        recip_cb_desc.total_size = cb_config.reciprocal_CB_size_bytes;
        recip_cb_desc.core_ranges = core_ranges.all_cores;
        recip_cb_desc.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_25,
            .data_format = cb_config.reciprocal_cb_data_format,
            .page_size = cb_config.reciprocal_CB_size_bytes});
        recip_cb_desc.buffer = cb_config.recip_buffer;
        program_descriptor.cbs.push_back(std::move(recip_cb_desc));
    }

    if (cb_config.is_post_all_gather) {
        // CB 7: cb_stats
        CBDescriptor stats_cb_desc;
        stats_cb_desc.total_size = cb_config.stats_cb_size;
        stats_cb_desc.core_ranges = core_ranges.sender_cores;
        stats_cb_desc.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_7,
            .data_format = cb_config.cb_data_format,
            .page_size = cb_config.single_tile_size});
        stats_cb_desc.buffer = cb_config.stats_buffer;
        program_descriptor.cbs.push_back(std::move(stats_cb_desc));

        // CB 21: cb_stats_reduced
        program_descriptor.cbs.push_back(make_cb_descriptor(
            cb_config.stats_reduced_cb_size,
            core_ranges.sender_cores,
            tt::CBIndex::c_21,
            cb_config.cb_data_format,
            cb_config.single_tile_size));

        // CB 19: cb_var
        program_descriptor.cbs.push_back(make_cb_descriptor(
            cb_config.ex_global_CB_size,
            core_ranges.sender_cores,
            tt::CBIndex::c_19,
            cb_config.cb_data_format,
            cb_config.single_tile_size));
    }

    // CB 16: output
    if (cb_config.is_pre_all_gather) {
        program_descriptor.cbs.push_back(make_cb_descriptor(
            cb_config.out_CB_size,
            core_ranges.sender_cores,
            tt::CBIndex::c_16,
            cb_config.out_data_format,
            cb_config.out_single_tile_size,
            cb_config.output_buffer));
    } else {
        program_descriptor.cbs.push_back(make_cb_descriptor(
            cb_config.out_CB_size,
            core_ranges.all_cores,
            tt::CBIndex::c_16,
            cb_config.out_data_format,
            cb_config.out_single_tile_size,
            cb_config.output_buffer));
    }

    // CB 17: output reshard (if is_post_all_gather and not skip_write_back)
    if (cb_config.is_post_all_gather && !cb_config.skip_write_back) {
        program_descriptor.cbs.push_back(make_cb_descriptor(
            cb_config.out_reshard_CB_size,
            all_worker_and_storage_cores,
            tt::CBIndex::c_17,
            cb_config.out_data_format,
            cb_config.out_single_tile_size,
            cb_config.output_reshard_buffer));
    }
}

//////////////////////////////////////////////////////////////////////////////
// Runtime args building
//////////////////////////////////////////////////////////////////////////////

CoreIndices CoreIndices::compute(uint32_t core_idx, const CoreCoord& core, const RuntimeArgsContext& ctx) {
    CoreIndices idx;

    if (ctx.grid.mcast_1d) {
        idx.height_index = 0;
        idx.width_index = core_idx;
    } else {
        if (ctx.grid.row_wise) {
            idx.height_index = core.y;
            idx.width_index = core.x;
        } else {
            idx.height_index = core.x;
            idx.width_index = core.y;
        }
    }

    idx.width_index_two_stage = idx.width_index % ctx.workers.num_blocks_first_stage;

    if (ctx.grid.use_two_stage_reduce) {
        idx.all_to_all_worker_tile_offset_bytes =
            (idx.width_index_two_stage * ctx.workers.num_rows_per_all_to_all_worker) * ctx.single_tile_size;
    } else {
        idx.all_to_all_worker_tile_offset_bytes =
            (idx.width_index * ctx.workers.num_rows_per_all_to_all_worker) * ctx.single_tile_size;
    }

    idx.gamma_tile_start_id = idx.width_index * ctx.block_wt;
    idx.beta_tile_start_id = idx.width_index * ctx.block_wt;

    idx.num_reduce_tiles_per_block_h = ctx.block_wt;
    if (idx.width_index == ctx.last_core_width_index) {
        idx.num_reduce_tiles_per_block_h = ctx.Kt - ctx.last_core_width_index * ctx.block_wt;
    }

    return idx;
}

bool CoreIndices::is_all_to_all(const RuntimeArgsContext& ctx) const {
    if (ctx.grid.use_two_stage_reduce) {
        return width_index_two_stage < ctx.workers.num_cores_all_to_all_first_stage;
    }
    return width_index < ctx.workers.num_cores_all_to_all;
}

std::vector<uint32_t> build_compute_args(
    const CoreIndices& idx, const RuntimeArgsContext& ctx, bool& is_all_to_all_out) {
    std::vector<uint32_t> args{idx.num_reduce_tiles_per_block_h};
    is_all_to_all_out = idx.is_all_to_all(ctx);

    if (is_all_to_all_out) {
        uint32_t num_rows;
        if (ctx.grid.use_two_stage_reduce) {
            num_rows = idx.width_index_two_stage == ctx.workers.num_cores_all_to_all_first_stage - 1
                           ? ctx.workers.num_rows_per_all_to_all_worker_last
                           : ctx.workers.num_rows_per_all_to_all_worker;
        } else {
            num_rows = idx.width_index == ctx.workers.num_cores_all_to_all - 1
                           ? ctx.workers.num_rows_per_all_to_all_worker_last
                           : ctx.workers.num_rows_per_all_to_all_worker;
        }
        args.push_back(num_rows);
        args.push_back((uint32_t)ctx.grid.use_two_stage_reduce);
        bool is_second_stage_reader =
            ctx.grid.use_two_stage_reduce && idx.width_index < ctx.workers.num_cores_all_to_all_first_stage;
        args.push_back((uint32_t)is_second_stage_reader);
        if (ctx.is_post_all_gather) {
            args.push_back((uint32_t)ctx.num_distributed_devices);
        }
    }
    return args;
}

std::vector<uint32_t> build_reader_sender_args(
    const CoreCoord& core, const CoreIndices& idx, const RuntimeArgsContext& ctx, IDevice* device) {
    // Compute mcast range
    CoreCoord mcast_start, mcast_end;
    if (ctx.grid.mcast_1d) {
        CoreCoord top_left = {(std::size_t)ctx.core_ranges.start_core.x, (std::size_t)ctx.core_ranges.start_core.y};
        CoreCoord bottom_right = {
            (std::size_t)ctx.core_ranges.start_core.x + ctx.grid.grid_size.x - 1,
            (std::size_t)ctx.core_ranges.start_core.y + ctx.grid.grid_size.y - 1};
        mcast_start = device->worker_core_from_logical_core(top_left);
        mcast_end = device->worker_core_from_logical_core(bottom_right);
    } else {
        if (ctx.grid.row_wise) {
            CoreCoord left_plus_one = {(std::size_t)ctx.core_ranges.start_core.x + 1, (std::size_t)core.y};
            CoreCoord right = {
                (std::size_t)ctx.core_ranges.start_core.x + ctx.grid.grid_size.x - 1, (std::size_t)core.y};
            mcast_start = device->worker_core_from_logical_core(left_plus_one);
            mcast_end = device->worker_core_from_logical_core(right);
        } else {
            CoreCoord top_plus_one = {(std::size_t)core.x, (std::size_t)ctx.core_ranges.start_core.y + 1};
            CoreCoord bottom = {
                (std::size_t)core.x, (std::size_t)ctx.core_ranges.start_core.y + ctx.grid.grid_size.y - 1};
            mcast_start = device->worker_core_from_logical_core(top_plus_one);
            mcast_end = device->worker_core_from_logical_core(bottom);
        }
    }
    if (ctx.reader_noc == NOC::NOC_1) {
        std::swap(mcast_start, mcast_end);
    }

    std::vector<uint32_t> args;
    args.push_back(mcast_start.x);
    args.push_back(mcast_start.y);
    args.push_back(mcast_end.x);
    args.push_back(mcast_end.y);

    if (ctx.grid.mcast_1d) {
        args.push_back(core.x - ctx.core_ranges.start_core.x);
        args.push_back(core.y - ctx.core_ranges.start_core.y);
        args.insert(args.end(), ctx.mcast_noc_x.begin(), ctx.mcast_noc_x.end());
        args.insert(args.end(), ctx.mcast_noc_y.begin(), ctx.mcast_noc_y.end());
    } else {
        if (ctx.grid.row_wise) {
            args.push_back(core.x - ctx.core_ranges.start_core.x);
            args.push_back(0);
            args.insert(args.end(), ctx.mcast_noc_x.begin(), ctx.mcast_noc_x.end());
            args.push_back(ctx.mcast_noc_y[idx.height_index]);
        } else {
            args.push_back(0);
            args.push_back(core.y - ctx.core_ranges.start_core.y);
            args.push_back(ctx.mcast_noc_x[idx.height_index]);
            args.insert(args.end(), ctx.mcast_noc_y.begin(), ctx.mcast_noc_y.end());
        }
    }
    return args;
}

std::vector<uint32_t> build_reader_receiver_all_to_all_args(
    const CoreCoord& core, const CoreIndices& idx, const RuntimeArgsContext& ctx) {
    std::vector<uint32_t> args;

    bool is_last_all_to_all_worker;
    if (ctx.grid.use_two_stage_reduce) {
        is_last_all_to_all_worker = idx.width_index_two_stage == ctx.workers.num_cores_all_to_all_first_stage - 1;
    } else {
        is_last_all_to_all_worker = idx.width_index == ctx.workers.num_cores_all_to_all - 1;
    }
    args.push_back(is_last_all_to_all_worker);
    args.push_back(idx.all_to_all_worker_tile_offset_bytes);

    bool is_second_stage_reader =
        ctx.grid.use_two_stage_reduce && idx.width_index < ctx.workers.num_cores_all_to_all_first_stage;
    args.push_back((uint32_t)is_second_stage_reader);

    if (ctx.grid.mcast_1d) {
        args.push_back(core.x - ctx.core_ranges.start_core.x);
        args.push_back(core.y - ctx.core_ranges.start_core.y);
        args.insert(args.end(), ctx.mcast_noc_x.begin(), ctx.mcast_noc_x.end());
        args.insert(args.end(), ctx.mcast_noc_y.begin(), ctx.mcast_noc_y.end());
    } else {
        if (ctx.grid.row_wise) {
            args.push_back(core.x - ctx.core_ranges.start_core.x);
            args.push_back(0);
            args.insert(args.end(), ctx.mcast_noc_x.begin(), ctx.mcast_noc_x.end());
            args.push_back(ctx.mcast_noc_y[idx.height_index]);
        } else {
            args.push_back(0);
            args.push_back(core.y - ctx.core_ranges.start_core.y);
            args.push_back(ctx.mcast_noc_x[idx.height_index]);
            args.insert(args.end(), ctx.mcast_noc_y.begin(), ctx.mcast_noc_y.end());
        }
    }
    return args;
}

std::vector<uint32_t> build_reader_receiver_not_all_to_all_args(const CoreIndices& idx, const RuntimeArgsContext& ctx) {
    std::vector<uint32_t> args;
    args.push_back(false);  // is_last_all_to_all_worker
    args.push_back(idx.all_to_all_worker_tile_offset_bytes);
    args.push_back(0);  // is_second_stage_reader
    args.push_back(0);
    args.push_back(0);

    if (ctx.grid.mcast_1d) {
        args.push_back(ctx.mcast_noc_x[0]);
        args.push_back(ctx.mcast_noc_y[0]);
    } else {
        if (ctx.grid.row_wise) {
            args.push_back(ctx.mcast_noc_x[0]);
            args.push_back(ctx.mcast_noc_y[idx.height_index]);
        } else {
            args.push_back(ctx.mcast_noc_x[idx.height_index]);
            args.push_back(ctx.mcast_noc_y[0]);
        }
    }
    return args;
}

std::vector<uint32_t> build_write_back_args(
    const RuntimeArgsContext& ctx, uint32_t& current_storage_core, uint32_t& current_storage_core_offset) {
    std::vector<uint32_t> args;
    if (!ctx.is_post_all_gather) {
        return args;
    }

    args.push_back(current_storage_core_offset * ctx.out_single_tile_size);
    uint32_t num_segments = 0;
    uint32_t worker_offset = 0;

    while (worker_offset < ctx.block_wt) {
        uint32_t tiles_available = ctx.block_wt_resharded - current_storage_core_offset;
        uint32_t tiles_left = ctx.block_wt - worker_offset;
        uint32_t tiles_to_write = std::min(tiles_left, tiles_available);

        num_segments += 1;
        args.push_back(tiles_to_write * ctx.out_single_tile_size);
        args.push_back(ctx.storage_core_noc_x[current_storage_core]);
        args.push_back(ctx.storage_core_noc_y[current_storage_core]);

        worker_offset += tiles_to_write;
        current_storage_core_offset += tiles_to_write;
        if (current_storage_core_offset >= ctx.block_wt_resharded) {
            current_storage_core += 1;
            current_storage_core_offset = 0;
            TT_FATAL(
                current_storage_core <= ctx.num_storage_cores,
                "current_storage_core {} is exceeding number of storage cores {}",
                current_storage_core,
                ctx.num_storage_cores);
        }
    }
    args.insert(args.begin(), num_segments);
    return args;
}

std::vector<uint32_t> build_writer_args(
    const CoreIndices& idx,
    const RuntimeArgsContext& ctx,
    const std::vector<uint32_t>& write_back_args,
    bool is_all_to_all) {
    std::vector<uint32_t> args;

    if (is_all_to_all) {
        if (ctx.grid.use_two_stage_reduce && idx.width_index >= ctx.workers.num_cores_all_to_all_first_stage) {
            args.push_back(ctx.packed_cinv_value_one);
        } else {
            args.push_back(ctx.packed_cinv_value);
        }
    } else {
        args.push_back(ctx.packed_cinv_value);
    }
    args.push_back(ctx.packed_winv_value);
    args.push_back(ctx.eps_u);
    args.push_back(ctx.gamma_dram_addr);
    args.push_back(ctx.beta_dram_addr);
    args.push_back(idx.gamma_tile_start_id);
    args.push_back(idx.beta_tile_start_id);
    args.insert(args.end(), write_back_args.begin(), write_back_args.end());
    return args;
}

RuntimeArgsResult RuntimeArgsResult::build(
    const std::vector<CoreCoord>& cores, RuntimeArgsContext& ctx, IDevice* device) {
    RuntimeArgsResult result;

    uint32_t current_storage_core = 0;
    uint32_t current_storage_core_offset = 0;

    for (uint32_t i = 0; i < cores.size(); ++i) {
        const auto& core = cores[i];
        auto idx = CoreIndices::compute(i, core, ctx);

        // Compute runtime args
        bool is_all_to_all = false;
        auto compute_args = build_compute_args(idx, ctx, is_all_to_all);
        if (is_all_to_all) {
            result.compute_all_to_all.emplace_back(core, compute_args);
        } else {
            result.compute_not_all_to_all.emplace_back(core, compute_args);
        }

        // Reader runtime args
        if (idx.width_index == 0) {
            auto reader_args = build_reader_sender_args(core, idx, ctx, device);
            result.reader_sender.emplace_back(core, reader_args);
        } else if (is_all_to_all) {
            auto reader_args = build_reader_receiver_all_to_all_args(core, idx, ctx);
            result.reader_receiver_all_to_all.emplace_back(core, reader_args);
        } else {
            auto reader_args = build_reader_receiver_not_all_to_all_args(idx, ctx);
            result.reader_receiver.emplace_back(core, reader_args);
        }

        // Writer runtime args
        auto write_back_args = build_write_back_args(ctx, current_storage_core, current_storage_core_offset);
        auto writer_args = build_writer_args(idx, ctx, write_back_args, is_all_to_all);
        if (is_all_to_all) {
            result.writer_sender.emplace_back(core, writer_args);
        } else {
            result.writer_receiver.emplace_back(core, writer_args);
        }
    }

    return result;
}

}  // namespace ttnn::prim::sharded_layernorm_helpers

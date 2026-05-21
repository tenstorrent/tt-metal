// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/normalization/layernorm/device/sharded_layernorm_factory_helpers.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim::sharded_layernorm_helpers {

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

std::tuple<tt::DataFormat, tt::DataFormat, tt::DataFormat, tt::DataFormat, tt::DataFormat, tt::DataFormat>
get_cb_data_formats(
    const Tensor& output,
    const std::optional<const Tensor>& gamma,
    const std::optional<const Tensor>& beta,
    const std::optional<const Tensor>& stats,
    bool fp32_dest_acc_en) {
    tt::DataFormat out_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    tt::DataFormat cb_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    tt::DataFormat gamma_cb_data_format = gamma.has_value()
                                              ? tt::tt_metal::datatype_to_dataformat_converter(gamma.value().dtype())
                                              : tt::DataFormat::Float16_b;
    tt::DataFormat beta_cb_data_format = beta.has_value()
                                             ? tt::tt_metal::datatype_to_dataformat_converter(beta.value().dtype())
                                             : tt::DataFormat::Float16_b;
    tt::DataFormat stats_cb_data_format = stats.has_value()
                                              ? tt::tt_metal::datatype_to_dataformat_converter(stats.value().dtype())
                                              : tt::DataFormat::Float16_b;
    tt::DataFormat reciprocal_cb_data_format = tt::DataFormat::Float32;
    return {
        out_data_format,
        cb_data_format,
        gamma_cb_data_format,
        beta_cb_data_format,
        stats_cb_data_format,
        reciprocal_cb_data_format};
}

namespace {

bool should_use_two_stage_reduce(
    bool mcast_1d, bool row_wise, CoreCoord grid_size, CoreCoord compute_with_storage_grid_size) {
    if (mcast_1d) {
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
}

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

}  // namespace

GridParams GridParams::compute(const Tensor& input, uint32_t block_ht, CoreCoord compute_with_storage_grid_size) {
    auto spec = input.shard_spec().value();
    const uint32_t tile_height = input.tensor_spec().tile().get_height();
    uint32_t M = input.physical_volume() / input.padded_shape()[-1];
    uint32_t block_h = block_ht * tile_height;
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

    if (grid.grid_offset.has_value()) {
        const auto& offset = grid.grid_offset.value();
        cr.start_core = {cr.start_core.x + offset.x, cr.start_core.y + offset.y};
        cr.sender_cores = {
            {cr.sender_cores.start_coord.x + offset.x, cr.sender_cores.start_coord.y + offset.y},
            {cr.sender_cores.end_coord.x + offset.x, cr.sender_cores.end_coord.y + offset.y}};
        cr.all_to_all_cores = apply_grid_offset(cr.all_to_all_cores, offset);
        cr.all_to_all_workers_except_sender = apply_grid_offset(cr.all_to_all_workers_except_sender, offset);
        cr.not_all_to_all_workers = apply_grid_offset(cr.not_all_to_all_workers, offset);
    }

    return cr;
}

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
    bool has_b,
    bool has_gamma,
    bool has_beta,
    bool rms_norm,
    bool use_welford,
    bool skip_write_back,
    const std::optional<operations::unary::UnaryWithParam>& fused_activation,
    std::optional<tt::tt_metal::DataType> output_dtype) {
    KernelDefines defines;

    if (has_b) {
        defines.reader.emplace_back("FUSE_PRE_ADD", "1");
    }
    if (has_gamma) {
        defines.reader.emplace_back("FUSE_GAMMA", "1");
    }
    if (has_beta) {
        defines.reader.emplace_back("FUSE_BETA", "1");
    }

    if (rms_norm) {
        defines.writer.emplace_back("RMSNORM", "1");
    }
    if (skip_write_back) {
        defines.writer.emplace_back("SKIP_WRITE_BACK", "1");
    }

    if (has_b) {
        defines.compute.emplace_back("FUSE_PRE_ADD", "1");
    }
    if (rms_norm && !use_welford) {
        defines.compute.emplace_back("RMSNORM", "1");
    }
    if (fused_activation.has_value()) {
        const auto& act = fused_activation.value();
        auto act_defines =
            ttnn::operations::unary::utils::get_defines(act.op_type, act.params, "ACTIVATION", "w", output_dtype);
        for (auto& [key, val] : act_defines) {
            defines.compute.emplace_back(key, val);
        }
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
    if (is_post_all_gather && !rms_norm) {
        sizes.x_CB_size += single_tile_size;
    }
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
        sizes.stats_cb_size = post_all_gather_stats_block_tiles * stats_single_tile_size;
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

PerCoreIndices PerCoreIndices::compute(
    uint32_t core_idx,
    const CoreCoord& core,
    const GridParams& grid,
    const WorkerDistribution& workers,
    uint32_t block_wt,
    uint32_t Kt,
    uint32_t last_core_width_index,
    uint32_t single_tile_size) {
    PerCoreIndices idx;

    if (grid.mcast_1d) {
        idx.height_index = 0;
        idx.width_index = core_idx;
    } else {
        CoreCoord offset = grid.grid_offset.value_or(CoreCoord{0, 0});
        if (grid.row_wise) {
            idx.height_index = core.y - offset.y;
            idx.width_index = core.x - offset.x;
        } else {
            idx.height_index = core.x - offset.x;
            idx.width_index = core.y - offset.y;
        }
    }

    idx.width_index_two_stage = idx.width_index % workers.num_blocks_first_stage;

    if (grid.use_two_stage_reduce) {
        idx.all_to_all_worker_tile_offset_bytes =
            (idx.width_index_two_stage * workers.num_rows_per_all_to_all_worker) * single_tile_size;
    } else {
        idx.all_to_all_worker_tile_offset_bytes =
            (idx.width_index * workers.num_rows_per_all_to_all_worker) * single_tile_size;
    }

    idx.gamma_tile_start_id = idx.width_index * block_wt;
    idx.beta_tile_start_id = idx.width_index * block_wt;

    idx.num_reduce_tiles_per_block_h = block_wt;
    if (idx.width_index == last_core_width_index) {
        idx.num_reduce_tiles_per_block_h = Kt - last_core_width_index * block_wt;
    }

    return idx;
}

bool PerCoreIndices::is_all_to_all(const GridParams& grid, const WorkerDistribution& workers) const {
    if (grid.use_two_stage_reduce) {
        return width_index_two_stage < workers.num_cores_all_to_all_first_stage;
    }
    return width_index < workers.num_cores_all_to_all;
}

}  // namespace ttnn::prim::sharded_layernorm_helpers

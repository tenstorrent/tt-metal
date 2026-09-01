// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/normalization/layernorm/device/sharded_layernorm_factory_helpers.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"

#include <algorithm>
#include <bit>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim::sharded_layernorm_helpers {

//////////////////////////////////////////////////////////////////////////////
// Spec identities
//////////////////////////////////////////////////////////////////////////////

const m2::KernelSpecName READER_SENDER{"reader_sender"};
const m2::KernelSpecName READER_RECEIVER_ALL_TO_ALL{"reader_receiver_all_to_all"};
const m2::KernelSpecName READER_RECEIVER{"reader_receiver"};
const m2::KernelSpecName WRITER_SENDER{"writer_sender"};
const m2::KernelSpecName WRITER_RECEIVER{"writer_receiver"};
const m2::KernelSpecName COMPUTE_ALL_TO_ALL{"compute_all_to_all"};
const m2::KernelSpecName COMPUTE_NOT_ALL_TO_ALL{"compute_not_all_to_all"};
const m2::KernelSpecName IDLE_READER{"idle_reader"};
const m2::KernelSpecName IDLE_WRITER{"idle_writer"};
const m2::KernelSpecName IDLE_COMPUTE{"idle_compute"};

const m2::DFBSpecName IN0{"in0"};
const m2::DFBSpecName IN1{"in1"};
const m2::DFBSpecName IN_PRE_ADD{"in_pre_add"};
const m2::DFBSpecName SCALER{"scaler"};
const m2::DFBSpecName EPS{"eps"};
const m2::DFBSpecName SCALER_GLOBAL{"scaler_global"};
const m2::DFBSpecName GAMMA{"gamma"};
const m2::DFBSpecName BETA{"beta"};
const m2::DFBSpecName STATS{"stats"};
const m2::DFBSpecName EX_PARTIAL{"ex_partial"};
const m2::DFBSpecName EX{"ex"};
const m2::DFBSpecName EX_EXTERNAL{"ex_external"};
const m2::DFBSpecName EX_PARTIAL2{"ex_partial2"};
const m2::DFBSpecName EX2{"ex2"};
const m2::DFBSpecName EX_EXTERNAL2{"ex_external2"};
const m2::DFBSpecName MASK_SCRATCH{"mask_scratch"};
const m2::DFBSpecName EX_GLOBAL{"ex_global"};
const m2::DFBSpecName OUT{"out"};
const m2::DFBSpecName XMM{"xmm"};
const m2::DFBSpecName COL_MASK{"col_mask"};
const m2::DFBSpecName VAR{"var"};
const m2::DFBSpecName EX2PE{"ex2pe"};
const m2::DFBSpecName STATS_REDUCED{"stats_reduced"};
const m2::DFBSpecName TRANSPOSE{"transpose"};
const m2::DFBSpecName X{"x"};
const m2::DFBSpecName RECIPROCALS{"reciprocals"};
const m2::DFBSpecName X_WELFORD{"x_welford"};

const m2::TensorParamName INPUT{"input"};
const m2::TensorParamName RESIDUAL{"residual"};
const m2::TensorParamName GAMMA_T{"weight"};
const m2::TensorParamName BETA_T{"bias"};
const m2::TensorParamName STATS_T{"stats"};
const m2::TensorParamName RECIP{"recip"};
const m2::TensorParamName OUTPUT{"output"};

const m2::SemaphoreSpecName REDUCE_SENDER{"reduce_sender"};
const m2::SemaphoreSpecName REDUCE_RECEIVER{"reduce_receiver"};
const m2::SemaphoreSpecName REDUCE_SECOND_STAGE{"reduce_second_stage"};

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

std::tuple<tt::DataFormat, tt::DataFormat, tt::DataFormat, tt::DataFormat, tt::DataFormat, tt::DataFormat>
get_dfb_data_formats(
    const Tensor& output,
    const std::optional<const Tensor>& gamma,
    const std::optional<const Tensor>& beta,
    const std::optional<const Tensor>& stats,
    bool fp32_dest_acc_en) {
    tt::DataFormat out_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    tt::DataFormat dfb_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    tt::DataFormat gamma_dfb_data_format = gamma.has_value()
                                               ? tt::tt_metal::datatype_to_dataformat_converter(gamma.value().dtype())
                                               : tt::DataFormat::Float16_b;
    tt::DataFormat beta_dfb_data_format = beta.has_value()
                                              ? tt::tt_metal::datatype_to_dataformat_converter(beta.value().dtype())
                                              : tt::DataFormat::Float16_b;
    tt::DataFormat stats_dfb_data_format = stats.has_value()
                                               ? tt::tt_metal::datatype_to_dataformat_converter(stats.value().dtype())
                                               : tt::DataFormat::Float16_b;
    tt::DataFormat reciprocal_dfb_data_format = tt::DataFormat::Float32;
    return {
        out_data_format,
        dfb_data_format,
        gamma_dfb_data_format,
        beta_dfb_data_format,
        stats_dfb_data_format,
        reciprocal_dfb_data_format};
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
    bool rectangular = spec.grid.num_cores() == gs.x * gs.y;
    bool two_stage = rectangular && should_use_two_stage_reduce(mcast, rw, gs, compute_with_storage_grid_size);
    return GridParams{
        .shard_spec = spec,
        .grid_size = gs,
        .grid_offset = offset,
        .mcast_1d = mcast,
        .row_wise = rw,
        .num_blocks = nb,
        .use_mcast = nb > 1,
        .use_two_stage_reduce = two_stage,
        .grid_is_rectangular = rectangular};
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

    const CoreRange bbox = grid.shard_spec.grid.bounding_box();
    cr.mcast_dest_cores = CoreRangeSet(bbox);
    cr.inactive_cores = cr.mcast_dest_cores.subtract(cr.all_cores);
    cr.num_mcast_dests = (grid.mcast_1d ? bbox.size() : grid.num_blocks) - 1;

    return cr;
}

//////////////////////////////////////////////////////////////////////////////
// Kernel paths and dataflow buffer sizes
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

DFBSizeParams::Sizes DFBSizeParams::compute() const {
    Sizes sizes;

    uint32_t in0_block_tiles = block_wt * block_ht;

    sizes.in0_dfb_size = in0_block_tiles * in_single_tile_size;
    sizes.in1_dfb_size = sizes.in0_dfb_size;
    sizes.in2_dfb_size = bfloat16_tile_size;
    sizes.in3_dfb_size = bfloat16_tile_size;
    sizes.in5_dfb_size = in0_block_tiles * gamma_single_tile_size / block_ht;
    sizes.in6_dfb_size = in0_block_tiles * beta_single_tile_size / block_ht;

    sizes.x_dfb_size = in0_block_tiles * single_tile_size;
    if (is_post_all_gather && !rms_norm) {
        // Non-RMSNORM post-allgather reuses x as both the E[x^2] and the intermediate buffer.
        // The allgather worker writes 1 tile to the E[x^2] slot first, advancing the write
        // pointer. The buffer needs an extra tile so the subsequent intermediate write has enough
        // contiguous space.
        sizes.x_dfb_size += single_tile_size;
    }
    sizes.xmm_dfb_size = in0_block_tiles * single_tile_size;

    sizes.ex_partial_dfb_size = in0_block_tiles * single_tile_size / block_wt;
    sizes.ex_external_dfb_size = tt::div_up(Kt, block_wt) * single_tile_size;

    if (is_pre_all_gather || is_post_all_gather) {
        sizes.ex_partial_dfb_size = sizes.ex_partial_dfb_size * pre_all_gather_stats_block_tiles;
    }

    sizes.ex_dfb_size = sizes.ex_partial_dfb_size;
    sizes.ex_global_dfb_size = sizes.ex_partial_dfb_size;
    sizes.ex2pe_dfb_size = num_rows_per_all_to_all_worker * single_tile_size;

    if (is_post_all_gather) {
        sizes.stats_dfb_size = post_all_gather_stats_block_tiles * stats_single_tile_size;
        sizes.stats_reduced_dfb_size = pre_all_gather_stats_block_tiles * single_tile_size;
    }

    if (is_pre_all_gather) {
        sizes.out_dfb_size = pre_all_gather_stats_block_tiles * out_single_tile_size;
    } else {
        sizes.out_dfb_size = in0_block_tiles * out_single_tile_size;
    }

    sizes.out_reshard_dfb_size = sizes.out_dfb_size;
    if (is_post_all_gather && !skip_write_back) {
        sizes.out_reshard_dfb_size = block_wt_resharded * block_ht * out_single_tile_size;
    }

    // Update ex_external_dfb_size based on configuration
    if (use_two_stage_reduce) {
        sizes.ex_external_dfb_size = (num_blocks_first_stage + num_blocks_second_stage - 1) * single_tile_size;
    }
    if (is_pre_all_gather) {
        sizes.ex_external_dfb_size = sizes.ex_external_dfb_size * pre_all_gather_stats_block_tiles;
    }

    if (use_welford) {
        sizes.ex_external_dfb_size *= 2;
        sizes.ex_partial_dfb_size *= 2;
        sizes.ex_dfb_size *= 2;
        sizes.ex_global_dfb_size *= 2;
    }

    return sizes;
}

//////////////////////////////////////////////////////////////////////////////
// Dataflow buffer specs
//////////////////////////////////////////////////////////////////////////////

namespace {

// A dataflow buffer's total size in Metal 2.0 is entry_size * num_entries. Every buffer in this
// factory is sized in bytes by DFBSizeParams and paged by one tile, so the entry count follows from
// the two. The single-entry form covers the buffers whose page size *is* their whole size.
void add_dfb(
    m2::ProgramSpec& spec,
    const m2::DFBSpecName& unique_id,
    uint32_t total_size,
    uint32_t entry_size,
    tt::DataFormat data_format,
    std::optional<m2::TensorParamName> borrowed_from = std::nullopt) {
    TT_FATAL(entry_size > 0, "Dataflow buffer '{}' has a zero entry size", unique_id);
    spec.dataflow_buffers.push_back(m2::DataflowBufferSpec{
        .unique_id = unique_id,
        .entry_size = entry_size,
        .num_entries = total_size / entry_size,
        .data_format_metadata = data_format,
        .borrowed_from = std::move(borrowed_from),
    });
}

// Make the two members of a legacy two-index circular buffer into an alias clique. Both members
// must name each other, which is what makes the group legal.
void alias_pair(m2::ProgramSpec& spec, const m2::DFBSpecName& first, const m2::DFBSpecName& second) {
    for (auto& dfb : spec.dataflow_buffers) {
        if (dfb.unique_id == first) {
            dfb.advanced_options.alias_with = {second};
        } else if (dfb.unique_id == second) {
            dfb.advanced_options.alias_with = {first};
        }
    }
}

std::optional<tt::DataFormat> data_format_of(const m2::ProgramSpec& spec, const m2::DFBSpecName& dfb) {
    for (const auto& candidate : spec.dataflow_buffers) {
        if (candidate.unique_id == dfb) {
            return candidate.data_format_metadata;
        }
    }
    return std::nullopt;
}

}  // namespace

void add_dataflow_buffer_specs(m2::ProgramSpec& spec, const SpecConfig& c) {
    const bool nd = !c.is_pre_all_gather && !c.is_post_all_gather;
    const auto& sizes = c.sizes;

    // Input shard. The compute kernel reads it in place; nothing streams into it.
    add_dfb(spec, IN0, sizes.in0_dfb_size, c.in_single_tile_size, c.in_data_format, INPUT);

    // Residual shard for the fused pre-add. The post-all-gather compute kernel has no pre-add, so it
    // never reads a residual even when one is supplied.
    if (c.has_b && !c.is_post_all_gather) {
        add_dfb(spec, IN1, sizes.in1_dfb_size, c.in_single_tile_size, c.in_data_format, RESIDUAL);
    }

    // Pre-all-gather pre-add destination. It borrows the *input* tensor, so a + b is written back
    // over a's own shard.
    if (c.is_pre_all_gather && c.has_b) {
        add_dfb(spec, IN_PRE_ADD, sizes.in1_dfb_size, c.in_single_tile_size, c.in_data_format, INPUT);
    }

    if (!c.use_welford) {
        add_dfb(spec, SCALER, sizes.in2_dfb_size, c.bfloat16_tile_size, tt::DataFormat::Float16_b);

        // The pre-all-gather compute kernel folds epsilon into the post-all-gather stage instead, so
        // it never reads an epsilon tile.
        if (!c.is_pre_all_gather) {
            add_dfb(spec, EPS, sizes.in3_dfb_size, c.bfloat16_tile_size, tt::DataFormat::Float16_b);
        }

        // Global reduce scaler: Float32 when the intermediates are Float32, otherwise bfloat16.
        const tt::DataFormat scaler_global_format =
            c.dfb_data_format == tt::DataFormat::Float32 ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
        const uint32_t scaler_global_tile_size = tt::tile_size(scaler_global_format);
        add_dfb(spec, SCALER_GLOBAL, scaler_global_tile_size, scaler_global_tile_size, scaler_global_format);

        // Scratch holding the masked input for the LayerNorm E[x] reduction, so the input buffer stays
        // intact for the (x - E[x]) pass. The mask itself is the writer-generated COL_MASK below.
        if (c.do_legacy_layernorm_col_mask) {
            add_dfb(spec, MASK_SCRATCH, sizes.xmm_dfb_size, c.single_tile_size, c.dfb_data_format);
        }
        if (c.do_col_mask) {
            // Writer-generated column mask, block_wt tiles (one tile-row), always in bfloat16: the mask
            // holds only 1.0 or 0.0, which is exact in that format. The writer fills it per core from
            // the core's width position; compute waits on it and reads by tile index.
            add_dfb(spec, COL_MASK, c.col_mask_gen_dfb_size_bytes, c.bfloat16_tile_size, tt::DataFormat::Float16_b);
        }

        // The Var[x] reduce chain. Only the non-distributed stage runs the second half of it on-core;
        // after the all-gather the partials arrive already reduced.
        if (!c.is_post_all_gather) {
            add_dfb(spec, EX_PARTIAL2, sizes.ex_partial_dfb_size, c.single_tile_size, c.dfb_data_format);
            add_dfb(spec, EX_EXTERNAL2, sizes.ex_external_dfb_size, c.single_tile_size, c.dfb_data_format);
        }
        add_dfb(spec, EX2, sizes.ex_dfb_size, c.single_tile_size, c.dfb_data_format);
        if (nd) {
            add_dfb(spec, EX2PE, sizes.ex2pe_dfb_size, c.single_tile_size, c.dfb_data_format);
        }
    }

    // The E[x] reduce chain. RMSNorm normalizes by the mean of squares and has no mean to reduce, and
    // the distributed stages carry E[x] through the statistics tensor rather than these buffers.
    if (!c.rms_norm && nd) {
        add_dfb(spec, EX_PARTIAL, sizes.ex_partial_dfb_size, c.single_tile_size, c.dfb_data_format);
        add_dfb(spec, EX, sizes.ex_dfb_size, c.single_tile_size, c.dfb_data_format);
        add_dfb(spec, EX_EXTERNAL, sizes.ex_external_dfb_size, c.single_tile_size, c.dfb_data_format);
    }

    if (c.has_gamma && !c.is_pre_all_gather) {
        add_dfb(spec, GAMMA, sizes.in5_dfb_size, c.gamma_single_tile_size, c.gamma_dfb_data_format);
    }
    if (c.has_beta && !c.is_pre_all_gather) {
        add_dfb(spec, BETA, sizes.in6_dfb_size, c.beta_single_tile_size, c.beta_dfb_data_format);
    }

    // x: the pre-add result before the all-gather, x itself in the middle stage, E[x]^2 after it.
    add_dfb(spec, X, sizes.x_dfb_size, c.single_tile_size, c.dfb_data_format);

    // x - E[x], and the gamma/beta streaming intermediate. The pre-all-gather stage produces
    // statistics only and never forms either.
    if (!c.is_pre_all_gather) {
        add_dfb(spec, XMM, sizes.xmm_dfb_size, c.single_tile_size, c.dfb_data_format);
    }

    // The multicast landing buffer for the final statistics. The pre-all-gather stage sends its
    // statistics off-device instead of broadcasting them back.
    if (!c.is_pre_all_gather) {
        add_dfb(spec, EX_GLOBAL, sizes.ex_global_dfb_size, c.single_tile_size, c.dfb_data_format);
    }

    if (c.use_welford) {
        // transpose_dest is currently unusable, so the Welford statistics are transposed back to
        // columns through this buffer instead.
        add_dfb(spec, TRANSPOSE, sizes.ex_global_dfb_size, c.single_tile_size, c.dfb_data_format);
        add_dfb(
            spec,
            RECIPROCALS,
            c.reciprocal_dfb_size_bytes,
            c.reciprocal_dfb_size_bytes,
            c.reciprocal_dfb_data_format,
            RECIP);
    }

    // Output. When the writer reshards, this buffer is a plain intermediate and the output tensor is
    // reached through its own binding instead.
    add_dfb(
        spec,
        OUT,
        sizes.out_dfb_size,
        c.out_single_tile_size,
        c.out_data_format,
        c.writes_back ? std::nullopt : std::optional<m2::TensorParamName>(OUTPUT));

    // Welford-fp32 alias: a second index over the primary buffer's SRAM, same total size. Alias group
    // members must agree on whether they borrow their memory, so the non-fused alias borrows the input
    // tensor exactly as the buffer it shares does.
    if (c.welford_fp32_alias) {
        if (c.has_b) {
            add_dfb(spec, X_WELFORD, sizes.x_dfb_size, c.single_tile_size, c.dfb_data_format);
            alias_pair(spec, X, X_WELFORD);
        } else {
            add_dfb(spec, X_WELFORD, sizes.in0_dfb_size, c.in_single_tile_size, c.in_data_format, INPUT);
            alias_pair(spec, IN0, X_WELFORD);
        }
    }

    // These three are the only buffers this op places on a subset of its cores: the all-to-all compute
    // kernel binds them alone, so they exist on the all-to-all cores only. A buffer's device-facing slot
    // is the lowest one no buffer sharing a core with it has taken, so declaring them here, after every
    // all-core buffer, keeps each core's occupied slots a gap-free run starting at zero.
    if (c.is_post_all_gather) {
        add_dfb(spec, STATS, sizes.stats_dfb_size, c.stats_single_tile_size, c.stats_dfb_data_format, STATS_T);
        add_dfb(spec, STATS_REDUCED, sizes.stats_reduced_dfb_size, c.single_tile_size, c.dfb_data_format);
        add_dfb(spec, VAR, sizes.ex_global_dfb_size, c.single_tile_size, c.dfb_data_format);
    }
}

void add_tensor_parameter_specs(
    m2::ProgramSpec& spec,
    const SpecConfig& c,
    const Tensor& input,
    const std::optional<Tensor>& residual,
    const std::optional<Tensor>& gamma,
    const std::optional<Tensor>& beta,
    const std::optional<Tensor>& stats,
    const std::optional<Tensor>& recip,
    const Tensor& output) {
    spec.tensor_parameters.push_back(m2::TensorParameter{.unique_id = INPUT, .spec = input.tensor_spec()});
    spec.tensor_parameters.push_back(m2::TensorParameter{.unique_id = OUTPUT, .spec = output.tensor_spec()});
    if (c.has_b && !c.is_post_all_gather) {
        spec.tensor_parameters.push_back(
            m2::TensorParameter{.unique_id = RESIDUAL, .spec = residual.value().tensor_spec()});
    }
    if (c.has_gamma && !c.is_pre_all_gather) {
        spec.tensor_parameters.push_back(
            m2::TensorParameter{.unique_id = GAMMA_T, .spec = gamma.value().tensor_spec()});
    }
    if (c.has_beta && !c.is_pre_all_gather) {
        spec.tensor_parameters.push_back(m2::TensorParameter{.unique_id = BETA_T, .spec = beta.value().tensor_spec()});
    }
    if (c.is_post_all_gather) {
        spec.tensor_parameters.push_back(
            m2::TensorParameter{.unique_id = STATS_T, .spec = stats.value().tensor_spec()});
    }
    if (c.use_welford) {
        spec.tensor_parameters.push_back(m2::TensorParameter{.unique_id = RECIP, .spec = recip.value().tensor_spec()});
    }
}

//////////////////////////////////////////////////////////////////////////////
// Kernel specs
//////////////////////////////////////////////////////////////////////////////

namespace {

void bind_dfb(m2::KernelSpec& kernel, const m2::DFBSpecName& dfb, std::string accessor_name, m2::DFBEndpointType role) {
    kernel.dfb_bindings.push_back(m2::DFBBinding{
        .dfb_spec_name = dfb,
        .accessor_name = std::move(accessor_name),
        .endpoint_type = role,
    });
}

// Bind a buffer the kernel both fills and drains. One accessor name serves both directions, so
// the kernel builds a single DataflowBuffer object for the FIFO.
void bind_self_loop(m2::KernelSpec& kernel, const m2::DFBSpecName& dfb, std::string accessor_name) {
    bind_dfb(kernel, dfb, accessor_name, m2::DFBEndpointType::PRODUCER);
    bind_dfb(kernel, dfb, std::move(accessor_name), m2::DFBEndpointType::CONSUMER);
}

void bind_tensor(m2::KernelSpec& kernel, const m2::TensorParamName& tensor, std::string accessor_name) {
    kernel.tensor_bindings.push_back(m2::TensorBinding{
        .tensor_parameter_name = tensor,
        .accessor_name = std::move(accessor_name),
    });
}

void bind_semaphore(m2::KernelSpec& kernel, const m2::SemaphoreSpecName& sem, std::string accessor_name) {
    kernel.semaphore_bindings.push_back(m2::SemaphoreBinding{
        .semaphore_spec_name = sem,
        .accessor_name = std::move(accessor_name),
    });
}

// Preprocessor flags shared by every kernel role. Each one gates a buffer that is absent in some
// configurations, so it has to reach the preprocessor: an `if constexpr` still looks up the binding
// token in its discarded branch.
void add_shared_defines(m2::KernelSpec& kernel, const SpecConfig& c) {
    if (c.has_b) {
        kernel.compiler_options.defines.emplace("FUSE_PRE_ADD", "1");
    }
    if (c.has_gamma) {
        kernel.compiler_options.defines.emplace("FUSE_GAMMA", "1");
    }
    if (c.has_beta) {
        kernel.compiler_options.defines.emplace("FUSE_BETA", "1");
    }
}

//--------------------------------------------------------------------------
// Reader
//--------------------------------------------------------------------------

m2::KernelSpec::CompileTimeArgs reader_sender_compile_time_args(
    const GridParams& grid, const WorkerDistribution& workers, const CoreRanges& core_ranges, const SpecConfig& c) {
    return {
        {"num_blocks", grid.num_blocks},
        {"block_h", c.block_ht},
        {"num_all_to_all_workers_first_stage", workers.num_cores_all_to_all_first_stage},
        {"num_tiles_per_worker", workers.num_rows_per_all_to_all_worker},
        {"num_tiles_per_worker_bytes", workers.num_rows_per_all_to_all_worker * c.single_tile_size},
        {"num_tiles_per_worker_last", workers.num_rows_per_all_to_all_worker_last},
        {"num_tiles_per_worker_last_bytes", workers.num_rows_per_all_to_all_worker_last * c.single_tile_size},
        {"row_major", static_cast<uint32_t>(grid.row_wise)},
        {"num_x", core_ranges.num_cores_x_mcast},
        {"num_y", core_ranges.num_cores_y_mcast},
        {"use_two_stage_reduce", static_cast<uint32_t>(grid.use_two_stage_reduce)},
        {"num_blocks_first_stage", workers.num_blocks_first_stage},
        {"num_blocks_second_stage", workers.num_blocks_second_stage},
        {"num_mcast_dests", core_ranges.num_mcast_dests},
    };
}

// The receiver kernel runs both on the all-to-all workers and on the workers that only wait for the
// multicast. The two differ in `is_all_to_all_worker` and, for the workers that never gather, in the
// multicast grid dimensions, which legacy pinned to 1 x 1 so the coordinate block shrinks to the one
// sender coordinate.
m2::KernelSpec::CompileTimeArgs reader_receiver_compile_time_args(
    const GridParams& grid,
    const WorkerDistribution& workers,
    const CoreRanges& core_ranges,
    const SpecConfig& c,
    bool is_all_to_all_worker) {
    return {
        {"num_blocks", grid.num_blocks},
        {"block_h", c.block_ht},
        {"is_all_to_all_worker", static_cast<uint32_t>(is_all_to_all_worker)},
        {"num_all_to_all_workers", workers.num_cores_all_to_all_first_stage},
        {"num_tiles_per_worker", workers.num_rows_per_all_to_all_worker},
        {"num_tiles_per_worker_last", workers.num_rows_per_all_to_all_worker_last},
        {"row_major", static_cast<uint32_t>(grid.row_wise)},
        {"num_x", is_all_to_all_worker ? core_ranges.num_cores_x_mcast : 1},
        {"num_y", is_all_to_all_worker ? core_ranges.num_cores_y_mcast : 1},
        {"use_two_stage_reduce", static_cast<uint32_t>(is_all_to_all_worker && grid.use_two_stage_reduce)},
        {"num_blocks_first_stage", is_all_to_all_worker ? workers.num_blocks_first_stage : 0},
        {"num_blocks_second_stage", is_all_to_all_worker ? workers.num_blocks_second_stage : 0},
    };
}

// `is_all_to_all_worker` is true for the reader instances placed on the all-to-all group: after the
// all-gather only those nodes carry the reduced-statistics buffer, so only they bind it.
void bind_reader_resources(m2::KernelSpec& kernel, const SpecConfig& c, bool is_all_to_all_worker) {
    bind_semaphore(kernel, REDUCE_RECEIVER, "reduce_receiver");
    bind_semaphore(kernel, REDUCE_SENDER, "reduce_sender");
    bind_semaphore(kernel, REDUCE_SECOND_STAGE, "reduce_second_stage");

    if (c.is_pre_all_gather) {
        // Before the all-gather the reader only serves the E[x^2] chain: it gathers the other cores'
        // partials and hands the combined result back to compute.
        bind_dfb(kernel, EX_PARTIAL2, "ex_partial2", m2::DFBEndpointType::CONSUMER);
        bind_dfb(kernel, EX_EXTERNAL2, "ex_external2", m2::DFBEndpointType::PRODUCER);
        bind_dfb(kernel, EX2, "ex2", m2::DFBEndpointType::CONSUMER);
        return;
    }
    if (c.is_post_all_gather) {
        // After the all-gather there is nothing to gather across cores: the sender multicasts the
        // reduced statistics and every reader receives them.
        bind_dfb(kernel, EX_GLOBAL, "ex_global", m2::DFBEndpointType::PRODUCER);
        if (is_all_to_all_worker) {
            bind_dfb(kernel, STATS_REDUCED, "stats_reduced", m2::DFBEndpointType::CONSUMER);
        }
        return;
    }

    if (!c.rms_norm) {
        bind_dfb(kernel, EX_PARTIAL, "ex_partial", m2::DFBEndpointType::CONSUMER);
        bind_dfb(kernel, EX_EXTERNAL, "ex_external", m2::DFBEndpointType::PRODUCER);
        bind_dfb(kernel, EX, "ex", m2::DFBEndpointType::CONSUMER);
    }
    if (!c.use_welford) {
        bind_dfb(kernel, EX_PARTIAL2, "ex_partial2", m2::DFBEndpointType::CONSUMER);
        bind_dfb(kernel, EX_EXTERNAL2, "ex_external2", m2::DFBEndpointType::PRODUCER);
        bind_dfb(kernel, EX2PE, "ex2pe", m2::DFBEndpointType::CONSUMER);
        bind_dfb(kernel, EX2, "ex2", m2::DFBEndpointType::CONSUMER);
    }
    bind_dfb(kernel, EX_GLOBAL, "ex_global", m2::DFBEndpointType::PRODUCER);
}

void add_reader_defines(m2::KernelSpec& kernel, const SpecConfig& c) {
    add_shared_defines(kernel, c);
    // Both flags decide which halves of the reduce chain exist, so they gate binding tokens rather
    // than arithmetic.
    if (c.rms_norm) {
        kernel.compiler_options.defines.emplace("RMSNORM", "1");
    }
    if (c.use_welford) {
        kernel.compiler_options.defines.emplace("USE_WELFORD", "1");
    }
}

//--------------------------------------------------------------------------
// Writer
//--------------------------------------------------------------------------

m2::KernelSpec::CompileTimeArgs writer_compile_time_args(
    const SpecConfig& c, bool is_all_to_all_worker, uint32_t writer_num_varargs) {
    m2::KernelSpec::CompileTimeArgs args{
        {"is_all_to_all_worker", static_cast<uint32_t>(is_all_to_all_worker)},
        {"block_w", c.block_wt},
    };
    if (c.do_col_mask) {
        args.emplace("logical_K", c.logical_K);
    }
    if (!c.is_pre_all_gather) {
        args.emplace("gamma_is_float32", static_cast<uint32_t>(c.gamma_dfb_data_format == tt::DataFormat::Float32));
        args.emplace("beta_is_float32", static_cast<uint32_t>(c.beta_dfb_data_format == tt::DataFormat::Float32));
    }
    if (c.writes_back) {
        args.emplace("worker_core_stride_w_bytes", c.block_wt * c.out_single_tile_size);
        args.emplace("storage_core_stride_w_bytes", c.block_wt_resharded * c.out_single_tile_size);
        args.emplace("block_ht", c.block_ht);
        // The segment block's length varies per node, and the kernel copies it into a local array, so
        // it needs a compile-time bound: the longest block any node was given. A zero bound would
        // give the kernel a zero-length array to copy into, so a caller that has not measured the
        // block yet is a construction error rather than a degenerate configuration.
        TT_FATAL(
            writer_num_varargs > 0,
            "Writer write-back segment block length is 0, but this configuration writes back. The "
            "length is measured by build_run_args, which must run before the kernel specs are built.");
        args.emplace("max_write_back_segments", writer_num_varargs / 3);
    }
    return args;
}

void bind_writer_resources(m2::KernelSpec& kernel, const SpecConfig& c) {
    if (c.is_pre_all_gather) {
        bind_dfb(kernel, SCALER, "scaler", m2::DFBEndpointType::PRODUCER);
        bind_dfb(kernel, SCALER_GLOBAL, "scaler_global", m2::DFBEndpointType::PRODUCER);
        if (c.do_col_mask) {
            bind_dfb(kernel, COL_MASK, "col_mask", m2::DFBEndpointType::PRODUCER);
        }
        return;
    }

    if (!c.use_welford) {
        if (c.is_post_all_gather) {
            // After the all-gather the compute kernel reduces the gathered statistics with the global
            // scaler alone, so the per-core scaler the writer still generates is never drained.
            bind_self_loop(kernel, SCALER, "scaler");
        } else {
            bind_dfb(kernel, SCALER, "scaler", m2::DFBEndpointType::PRODUCER);
        }
        bind_dfb(kernel, EPS, "eps", m2::DFBEndpointType::PRODUCER);
        bind_dfb(kernel, SCALER_GLOBAL, "scaler_global", m2::DFBEndpointType::PRODUCER);
        if (c.do_col_mask) {
            bind_dfb(kernel, COL_MASK, "col_mask", m2::DFBEndpointType::PRODUCER);
        }
    }
    if (c.has_gamma) {
        bind_dfb(kernel, GAMMA, "gamma", m2::DFBEndpointType::PRODUCER);
        bind_tensor(kernel, GAMMA_T, "gamma");
    }
    if (c.has_beta) {
        bind_dfb(kernel, BETA, "beta", m2::DFBEndpointType::PRODUCER);
        bind_tensor(kernel, BETA_T, "beta");
    }
    if (c.writes_back) {
        bind_dfb(kernel, OUT, "out", m2::DFBEndpointType::CONSUMER);
        // The write-back moves nothing through a buffer: it reads this binding's base address and
        // issues its own remote writes to the storage cores.
        bind_tensor(kernel, OUTPUT, "dst");
    }
}

void add_writer_defines(m2::KernelSpec& kernel, const SpecConfig& c) {
    add_shared_defines(kernel, c);
    if (c.rms_norm) {
        kernel.compiler_options.defines.emplace("RMSNORM", "1");
    }
    if (c.use_welford) {
        kernel.compiler_options.defines.emplace("USE_WELFORD", "1");
    }
    // The write-back reads runtime arguments only the post-all-gather stage supplies, so the build
    // that compiles it is the build where those arguments exist.
    if (!c.writes_back) {
        kernel.compiler_options.defines.emplace("SKIP_WRITE_BACK", "1");
    }
    if (c.do_col_mask) {
        kernel.compiler_options.defines.emplace("DO_COL_MASK", "1");
    }
}

//--------------------------------------------------------------------------
// Compute
//--------------------------------------------------------------------------

m2::KernelSpec::CompileTimeArgs compute_compile_time_args(
    const GridParams& grid, const WorkerDistribution& workers, const SpecConfig& c) {
    m2::KernelSpec::CompileTimeArgs args{
        {"num_blocks_first_stage", workers.num_blocks_first_stage},
        {"block_h", c.block_ht},
        {"block_w", c.block_wt},
        {"subblock_w", c.subblock_wt},
        {"num_subblocks_w", c.block_wt / c.subblock_wt},
        {"num_tiles_per_block", c.block_ht * c.block_wt},
        {"float32_dtype", static_cast<uint32_t>(c.fp32_dest_acc_en)},
        {"legacy_rsqrt", static_cast<uint32_t>(c.legacy_rsqrt)},
        {"num_blocks_second_stage", workers.num_blocks_second_stage},
    };

    if (c.use_welford) {
        // Number of valid (logical) columns in the final tile of the width. The kernel uses this
        // both to bound the partial Welford tile and, via last_block_w, to weight the final width
        // shard in the cross-core combine, so it must reflect the logical width, not padded K.
        const uint32_t last_tile_w = (c.logical_K % c.tile_width == 0) ? c.tile_width : (c.logical_K % c.tile_width);
        const uint32_t logical_Kt = (c.logical_K + c.tile_width - 1) / c.tile_width;
        // Number of valid (logical) tiles the final width block reduces. The other width blocks each own
        // block_wt tiles; the final block owns the remainder. Each block spans a whole number of tiles
        // (block_w columns), so when the logical width does not fill them evenly the final core owns
        // fewer than block_wt tiles, and the cross-core combine must weight it by its true width, not
        // block_w. A partial boundary tile is counted as a valid tile here; its valid-column count is
        // carried separately in last_tile_w and combined into last_block_w.
        // For example, w=96 gives 3 tiles, which sharded on two cores leaves two real tiles on the
        // first core and one real tile plus one padding tile on the second. For w=80 (also 3 tiles),
        // the second core owns last_block_wt = 1 tile that is itself partial (last_tile_w = 16 valid
        // columns) plus one padding tile.
        // The subtraction below does not underflow: validate_sharded_input requires the trailing width pad
        // to be strictly less than one shard, i.e. (num_blocks - 1) * block_wt < Kt, and the padded width
        // Kt equals logical_Kt (padded_shape rounds the logical width up to a whole tile). So
        // (num_blocks - 1) * block_wt is strictly less than logical_Kt, and the final width block always
        // owns at least one logical tile: last_block_wt >= 1.
        const uint32_t last_block_wt = logical_Kt - (grid.num_blocks - 1) * c.block_wt;

        args.emplace("tile_width", c.tile_width);
        args.emplace("last_tile_w", last_tile_w);
        args.emplace("W", c.logical_K);
        args.emplace("eps", std::bit_cast<uint32_t>(c.eps));
        args.emplace("per_core_recip_lut_size", c.per_core_recip_lut_size);
        args.emplace("last_block_wt", last_block_wt);
    }
    return args;
}

void bind_compute_resources(m2::KernelSpec& kernel, const SpecConfig& c, bool is_all_to_all_worker) {
    // Every stage reads its input shard in place, so compute owns both ends of it.
    bind_self_loop(kernel, IN0, "in0");
    if (c.has_b && !c.is_post_all_gather) {
        bind_self_loop(kernel, IN1, "in1");
    }
    if (c.has_gamma && !c.is_pre_all_gather) {
        bind_dfb(kernel, GAMMA, "gamma", m2::DFBEndpointType::CONSUMER);
    }
    if (c.has_beta && !c.is_pre_all_gather) {
        bind_dfb(kernel, BETA, "beta", m2::DFBEndpointType::CONSUMER);
    }
    bind_self_loop(kernel, X, "x");

    if (c.is_pre_all_gather) {
        if (c.has_b) {
            bind_self_loop(kernel, IN_PRE_ADD, "in_pre_add");
        }
        bind_dfb(kernel, SCALER, "scaler", m2::DFBEndpointType::CONSUMER);
        bind_dfb(kernel, SCALER_GLOBAL, "scaler_global", m2::DFBEndpointType::CONSUMER);
        bind_dfb(kernel, EX_PARTIAL2, "ex_partial2", m2::DFBEndpointType::PRODUCER);
        bind_dfb(kernel, EX_EXTERNAL2, "ex_external2", m2::DFBEndpointType::CONSUMER);
        if (c.do_col_mask) {
            bind_dfb(kernel, COL_MASK, "col_mask", m2::DFBEndpointType::CONSUMER);
        }
        // The combine writes its result into one of these two, and only the cores that gather run it.
        // Both endpoints are declared on every compute instance regardless: the buffers exist on every
        // node the reader spans, so each instance needs its producer side even where nothing writes it.
        // The statistics leave the device through the all-gather rather than through a kernel, so the
        // output has no reader and compute holds both of its ends.
        bind_dfb(kernel, EX2, "ex2", m2::DFBEndpointType::PRODUCER);
        bind_self_loop(kernel, OUT, "out");
        return;
    }

    bind_self_loop(kernel, XMM, "xmm");
    bind_dfb(kernel, EX_GLOBAL, "ex_global", m2::DFBEndpointType::CONSUMER);
    // Where the writer reshards, it drains this buffer; compute is then the producer alone. Otherwise
    // compute is its only toucher.
    if (c.writes_back) {
        bind_dfb(kernel, OUT, "out", m2::DFBEndpointType::PRODUCER);
    } else {
        bind_self_loop(kernel, OUT, "out");
    }

    if (c.is_post_all_gather) {
        bind_dfb(kernel, EPS, "eps", m2::DFBEndpointType::CONSUMER);
        bind_dfb(kernel, SCALER_GLOBAL, "scaler_global", m2::DFBEndpointType::CONSUMER);
        bind_self_loop(kernel, EX2, "ex2");
        // The statistics reduction runs only on the cores that gather, and its three buffers are
        // placed with it.
        if (is_all_to_all_worker) {
            bind_self_loop(kernel, STATS, "stats");
            bind_self_loop(kernel, VAR, "var");
            // The reader drains the reduced statistics to multicast them, so compute is the producer
            // alone even though it also reads its own result back.
            bind_dfb(kernel, STATS_REDUCED, "stats_reduced", m2::DFBEndpointType::PRODUCER);
        }
        return;
    }

    // Non-distributed. The reduce pipeline's buffers all leave this kernel for the reader, so compute
    // holds the producer side of each even where it reads its own result back on the way.
    if (!c.rms_norm) {
        bind_dfb(kernel, EX_PARTIAL, "ex_partial", m2::DFBEndpointType::PRODUCER);
        bind_dfb(kernel, EX, "ex", m2::DFBEndpointType::PRODUCER);
        bind_dfb(kernel, EX_EXTERNAL, "ex_external", m2::DFBEndpointType::CONSUMER);
    }
    if (c.use_welford) {
        bind_self_loop(kernel, TRANSPOSE, "transpose");
        bind_self_loop(kernel, RECIPROCALS, "reciprocals");
        if (c.welford_fp32_alias) {
            bind_self_loop(kernel, X_WELFORD, "x_welford");
        }
    } else {
        bind_dfb(kernel, SCALER, "scaler", m2::DFBEndpointType::CONSUMER);
        bind_dfb(kernel, EPS, "eps", m2::DFBEndpointType::CONSUMER);
        bind_dfb(kernel, SCALER_GLOBAL, "scaler_global", m2::DFBEndpointType::CONSUMER);
        bind_dfb(kernel, EX_PARTIAL2, "ex_partial2", m2::DFBEndpointType::PRODUCER);
        bind_dfb(kernel, EX2, "ex2", m2::DFBEndpointType::PRODUCER);
        bind_dfb(kernel, EX_EXTERNAL2, "ex_external2", m2::DFBEndpointType::CONSUMER);
        bind_dfb(kernel, EX2PE, "ex2pe", m2::DFBEndpointType::PRODUCER);
        if (c.do_legacy_layernorm_col_mask) {
            bind_self_loop(kernel, MASK_SCRATCH, "mask_scratch");
        }
        if (c.do_col_mask) {
            bind_dfb(kernel, COL_MASK, "col_mask", m2::DFBEndpointType::CONSUMER);
        }
    }
}

void add_compute_defines(m2::KernelSpec& kernel, const SpecConfig& c, bool is_all_to_all_worker) {
    add_shared_defines(kernel, c);
    if (c.rms_norm && !c.use_welford) {
        kernel.compiler_options.defines.emplace("RMSNORM", "1");
    }
    if (c.do_col_mask) {
        kernel.compiler_options.defines.emplace("DO_COL_MASK", "1");
    }
    if (c.welford_fp32_alias) {
        kernel.compiler_options.defines.emplace("WELFORD_FP32_ALIAS", "1");
    }
    // The all-to-all workers read three extra runtime arguments and touch three extra buffers, so the
    // distinction has to be visible to the preprocessor rather than to `if constexpr` alone.
    if (is_all_to_all_worker) {
        kernel.compiler_options.defines.emplace("IS_ALLGATHER_WORKER", "1");
    }
    for (const auto& [key, value] : c.activation_defines) {
        kernel.compiler_options.defines.emplace(key, value);
    }
}

// Legacy carried a vector indexed by buffer index, defaulted everywhere except the Welford alias.
// That one becomes UnpackToDest; the remaining Float32 buffers the kernel consumes get an explicit
// UnpackToSrc, which is required once the 32-bit Dest register is enabled and which legacy supplied
// silently.
//
// The choice of which buffers get which mode assumes a Gen1 target (Wormhole, Blackhole), where
// unpacking straight to Dest costs performance unless it is the only way to keep 32 bits of
// precision. That is why UnpackToDest appears only at the Welford alias and every other Float32
// buffer is pinned to UnpackToSrc. Gen2 reverses the tradeoff: unpacking to Dest is free there and
// is the preferred mode for anything the SFPU consumes, so these assignments stay legal but become
// slower than they need to be. They want revisiting before this op targets Gen2.
void set_compute_unpack_modes(m2::KernelSpec& kernel, const m2::ProgramSpec& spec, const SpecConfig& c) {
    auto& modes = std::get<m2::ComputeHardwareConfig>(kernel.hw_config).unpack_modes;
    if (c.welford_fp32_alias) {
        modes.emplace(X_WELFORD, UnpackMode::UnpackToDest);
    }
    if (!c.fp32_dest_acc_en) {
        return;
    }
    for (const auto& binding : kernel.dfb_bindings) {
        if (binding.endpoint_type != m2::DFBEndpointType::CONSUMER) {
            continue;
        }
        if (data_format_of(spec, binding.dfb_spec_name) != tt::DataFormat::Float32) {
            continue;
        }
        modes.emplace(binding.dfb_spec_name, UnpackMode::UnpackToSrc);
    }
}

}  // namespace

void add_kernel_and_work_unit_specs(
    m2::ProgramSpec& spec,
    const CoreRanges& core_ranges,
    const WorkerDistribution& workers,
    const GridParams& grid,
    const SpecConfig& c,
    uint32_t writer_num_varargs) {
    const bool has_reader_receiver_all_to_all = grid.use_mcast && !core_ranges.all_to_all_workers_except_sender.empty();
    const bool has_not_all_to_all_workers = workers.num_none_all_to_all_workers > 0;
    const bool has_inactive_cores = !core_ranges.inactive_cores.empty();

    const m2::DataMovementHardwareConfig reader_hw{
        .gen1_specific = m2::DataMovementHardwareConfig::DataMovement1XXConfig{
            .processor = DataMovementProcessor::RISCV_0, .noc = c.reader_noc, .noc_mode = NOC_MODE::DM_DEDICATED_NOC}};
    const m2::DataMovementHardwareConfig writer_hw{
        .gen1_specific = m2::DataMovementHardwareConfig::DataMovement1XXConfig{
            .processor = DataMovementProcessor::RISCV_1, .noc = c.writer_noc, .noc_mode = NOC_MODE::DM_DEDICATED_NOC}};

    // The reader's trailing coordinate block is one X coordinate per multicast column followed by one
    // Y coordinate per multicast row. Its length is a compile-time property of the kernel, but the
    // kernel indexes into it, so it stays a vararg block.
    const uint32_t sender_num_coords = core_ranges.num_cores_x_mcast + core_ranges.num_cores_y_mcast;

    //----------------------------------------------------------------------
    // Reader sender
    //----------------------------------------------------------------------
    m2::KernelSpec reader_sender{
        .unique_id = READER_SENDER,
        .source = c.reader_sender_path,
        .compile_time_args = reader_sender_compile_time_args(grid, workers, core_ranges, c),
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"mcast_dest_noc_start_x",
                  "mcast_dest_noc_start_y",
                  "mcast_dest_noc_end_x",
                  "mcast_dest_noc_end_y",
                  "start_x",
                  "start_y"}},
        .hw_config = reader_hw,
    };
    reader_sender.advanced_options.num_runtime_varargs = sender_num_coords;
    add_reader_defines(reader_sender, c);
    bind_reader_resources(reader_sender, c, /*is_all_to_all_worker=*/true);
    spec.kernels.push_back(std::move(reader_sender));

    //----------------------------------------------------------------------
    // Reader receivers
    //----------------------------------------------------------------------
    const m2::KernelSpec::RuntimeArgSchema receiver_schema{
        .runtime_arg_names = {
            "is_last_all_to_all_worker",
            "all_to_all_tile_offset_bytes",
            "is_second_stage_reader",
            "start_x",
            "start_y"}};

    auto make_reader_receiver = [&](const m2::KernelSpecName& name, bool is_all_to_all_worker) {
        m2::KernelSpec kernel{
            .unique_id = name,
            .source = c.reader_receiver_path,
            .compile_time_args = reader_receiver_compile_time_args(grid, workers, core_ranges, c, is_all_to_all_worker),
            .runtime_arg_schema = receiver_schema,
            .hw_config = reader_hw,
        };
        kernel.advanced_options.num_runtime_varargs = is_all_to_all_worker ? sender_num_coords : 2;
        add_reader_defines(kernel, c);
        bind_reader_resources(kernel, c, /*is_all_to_all_worker=*/is_all_to_all_worker);
        return kernel;
    };

    if (has_reader_receiver_all_to_all) {
        spec.kernels.push_back(make_reader_receiver(READER_RECEIVER_ALL_TO_ALL, /*is_all_to_all_worker=*/true));
    }
    if (has_not_all_to_all_workers) {
        spec.kernels.push_back(make_reader_receiver(READER_RECEIVER, /*is_all_to_all_worker=*/false));
    }

    //----------------------------------------------------------------------
    // Writers
    //----------------------------------------------------------------------
    // The pre-all-gather writer reads the packed scalers and, only when it generates a column mask,
    // this core's width offset. The other two additionally read epsilon, and the write-back build
    // reads the segment block's two scalars.
    auto writer_schema = [&]() {
        m2::KernelSpec::RuntimeArgSchema schema;
        if (!c.use_welford) {
            schema.runtime_arg_names.push_back("scalar_c");
            schema.runtime_arg_names.push_back("scalar_w");
            if (!c.is_pre_all_gather) {
                schema.runtime_arg_names.push_back("eps");
            }
        }
        if (!c.is_pre_all_gather || c.do_col_mask) {
            schema.runtime_arg_names.push_back("width_shard_tile_start_id");
        }
        if (c.writes_back) {
            schema.runtime_arg_names.push_back("num_segments_to_write_back");
            schema.runtime_arg_names.push_back("storage_core_start_offset");
        }
        return schema;
    }();

    auto make_writer = [&](const m2::KernelSpecName& name, bool is_all_to_all_worker, uint32_t num_varargs) {
        m2::KernelSpec kernel{
            .unique_id = name,
            .source = c.writer_path,
            .compile_time_args = writer_compile_time_args(c, is_all_to_all_worker, writer_num_varargs),
            .runtime_arg_schema = writer_schema,
            .hw_config = writer_hw,
        };
        kernel.advanced_options.num_runtime_varargs = num_varargs;
        add_writer_defines(kernel, c);
        bind_writer_resources(kernel, c);
        return kernel;
    };

    spec.kernels.push_back(make_writer(WRITER_SENDER, /*is_all_to_all_worker=*/true, writer_num_varargs));
    if (has_not_all_to_all_workers) {
        spec.kernels.push_back(make_writer(WRITER_RECEIVER, /*is_all_to_all_worker=*/false, writer_num_varargs));
    }

    //----------------------------------------------------------------------
    // Compute
    //----------------------------------------------------------------------
    auto compute_schema = [&](bool is_all_to_all_worker) {
        m2::KernelSpec::RuntimeArgSchema schema;
        schema.runtime_arg_names.push_back("num_reduce_tiles_per_block_h");
        if (is_all_to_all_worker) {
            schema.runtime_arg_names.push_back("num_rows_per_all_to_all_worker");
            schema.runtime_arg_names.push_back("use_two_stage_reduce");
            schema.runtime_arg_names.push_back("is_second_stage_reader");
            if (c.is_post_all_gather) {
                schema.runtime_arg_names.push_back("num_distributed_blocks");
            }
        }
        if (c.use_welford) {
            schema.runtime_arg_names.push_back("welford_reduce_w");
            if (is_all_to_all_worker) {
                schema.runtime_arg_names.push_back("boundary_width_index");
                schema.runtime_arg_names.push_back("my_width_index");
            }
        }
        return schema;
    };

    auto make_compute = [&](const m2::KernelSpecName& name, bool is_all_to_all_worker) {
        m2::KernelSpec kernel{
            .unique_id = name,
            .source = c.compute_path,
            // Legacy defaults opt_level on the per-kernel-type config struct, where a compute kernel
            // gets O3; Metal 2.0's single type-agnostic CompilerOptions defaults to O2 for both
            // kinds, so the level has to be stated to keep the compile and the link where they were.
            .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
            .compile_time_args = compute_compile_time_args(grid, workers, c),
            .runtime_arg_schema = compute_schema(is_all_to_all_worker),
            .hw_config = c.compute_hw,
        };
        add_compute_defines(kernel, c, is_all_to_all_worker);
        bind_compute_resources(kernel, c, is_all_to_all_worker);
        set_compute_unpack_modes(kernel, spec, c);
        return kernel;
    };

    spec.kernels.push_back(make_compute(COMPUTE_ALL_TO_ALL, /*is_all_to_all_worker=*/true));
    if (has_not_all_to_all_workers) {
        spec.kernels.push_back(make_compute(COMPUTE_NOT_ALL_TO_ALL, /*is_all_to_all_worker=*/false));
    }

    //----------------------------------------------------------------------
    // Idle triple
    //----------------------------------------------------------------------
    // A non-rectangular shard grid leaves holes inside the multicast bounding box. The reduction
    // multicasts across the whole box, so those nodes still need this program's dataflow buffers and
    // semaphores in place; they do no work of their own. Their kernels compile their bodies out
    // entirely, so they take no arguments — the buffers land on the node through the host-side
    // bindings alone.
    if (has_inactive_cores) {
        m2::KernelSpec idle_reader{
            .unique_id = IDLE_READER,
            .source = c.reader_receiver_path,
            .compile_time_args =
                reader_receiver_compile_time_args(grid, workers, core_ranges, c, /*is_all_to_all_worker=*/false),
            .hw_config = reader_hw,
        };
        add_reader_defines(idle_reader, c);
        idle_reader.compiler_options.defines.emplace("IDLE_CORE", "1");
        bind_reader_resources(idle_reader, c, /*is_all_to_all_worker=*/false);
        spec.kernels.push_back(std::move(idle_reader));

        m2::KernelSpec idle_writer{
            .unique_id = IDLE_WRITER,
            .source = c.writer_path,
            .compile_time_args = writer_compile_time_args(c, /*is_all_to_all_worker=*/false, writer_num_varargs),
            .hw_config = writer_hw,
        };
        add_writer_defines(idle_writer, c);
        idle_writer.compiler_options.defines.emplace("IDLE_CORE", "1");
        bind_writer_resources(idle_writer, c);
        spec.kernels.push_back(std::move(idle_writer));

        m2::KernelSpec idle_compute{
            .unique_id = IDLE_COMPUTE,
            .source = c.compute_path,
            .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
            .compile_time_args = compute_compile_time_args(grid, workers, c),
            .hw_config = c.compute_hw,
        };
        add_compute_defines(idle_compute, c, /*is_all_to_all_worker=*/false);
        idle_compute.compiler_options.defines.emplace("IDLE_CORE", "1");
        bind_compute_resources(idle_compute, c, /*is_all_to_all_worker=*/false);
        set_compute_unpack_modes(idle_compute, spec, c);
        spec.kernels.push_back(std::move(idle_compute));
    }

    //----------------------------------------------------------------------
    // Semaphores and work units
    //----------------------------------------------------------------------
    // The semaphore set spans the whole multicast bounding box, not just the active cores: the
    // reduction's multicast signals the holes too.
    for (const auto& name : {REDUCE_SENDER, REDUCE_RECEIVER, REDUCE_SECOND_STAGE}) {
        spec.semaphores.push_back(m2::SemaphoreSpec{.unique_id = name, .target_nodes = core_ranges.mcast_dest_cores});
    }

    // The writer and compute kernels of the all-to-all group span both the sender node set and the
    // rest of the all-to-all group, so they appear in two work units and their derived node set is
    // the union.
    spec.work_units.push_back(m2::WorkUnitSpec{
        .name = "sender",
        .kernels = {READER_SENDER, WRITER_SENDER, COMPUTE_ALL_TO_ALL},
        .target_nodes = core_ranges.sender_cores,
    });
    if (has_reader_receiver_all_to_all) {
        spec.work_units.push_back(m2::WorkUnitSpec{
            .name = "all_to_all_except_sender",
            .kernels = {READER_RECEIVER_ALL_TO_ALL, WRITER_SENDER, COMPUTE_ALL_TO_ALL},
            .target_nodes = core_ranges.all_to_all_workers_except_sender,
        });
    }
    if (has_not_all_to_all_workers) {
        spec.work_units.push_back(m2::WorkUnitSpec{
            .name = "not_all_to_all",
            .kernels = {READER_RECEIVER, WRITER_RECEIVER, COMPUTE_NOT_ALL_TO_ALL},
            .target_nodes = core_ranges.not_all_to_all_workers,
        });
    }
    if (has_inactive_cores) {
        spec.work_units.push_back(m2::WorkUnitSpec{
            .name = "inactive",
            .kernels = {IDLE_READER, IDLE_WRITER, IDLE_COMPUTE},
            .target_nodes = core_ranges.inactive_cores,
        });
    }
}

//////////////////////////////////////////////////////////////////////////////
// Runtime argument building
//////////////////////////////////////////////////////////////////////////////

CoreIndices CoreIndices::compute(uint32_t core_idx, const CoreCoord& core, const RuntimeArgsContext& ctx) {
    CoreIndices idx;

    if (ctx.grid.mcast_1d) {
        idx.height_index = 0;
        idx.width_index = core_idx;
    } else {
        // In the non-mcast 1d case, core coordinates come from the shard spec grid which already has
        // the grid offset embedded. Subtract it to get 0-based grid-relative indices.
        CoreCoord offset = ctx.grid.grid_offset.value_or(CoreCoord{0, 0});
        if (ctx.grid.row_wise) {
            idx.height_index = core.y - offset.y;
            idx.width_index = core.x - offset.x;
        } else {
            idx.height_index = core.x - offset.x;
            idx.width_index = core.y - offset.y;
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

    idx.width_shard_tile_start_id = idx.width_index * ctx.block_wt;

    idx.num_reduce_tiles_per_block_h = ctx.block_wt;
    if (idx.width_index == ctx.last_core_width_index) {
        idx.num_reduce_tiles_per_block_h = ctx.Kt - ctx.last_core_width_index * ctx.block_wt;
    }

    // Real (logical) column count this core reduces over (used by the Welford compute kernel, which has
    // no per-column mask). Cores before the last own a full block_w; the final real core owns the
    // remaining logical columns (which may end in a partial tile); any all-padding core beyond it owns
    // none. For a single width shard this is just the whole logical width.
    const uint32_t block_w = ctx.block_wt * TILE_WIDTH;
    if (idx.width_index < ctx.last_core_width_index) {
        idx.welford_reduce_w = block_w;
    } else if (idx.width_index == ctx.last_core_width_index) {
        idx.welford_reduce_w = ctx.logical_K - ctx.last_core_width_index * block_w;
    } else {
        idx.welford_reduce_w = 0;
    }

    return idx;
}

bool CoreIndices::is_all_to_all(const RuntimeArgsContext& ctx) const {
    if (ctx.grid.use_two_stage_reduce) {
        return width_index_two_stage < ctx.workers.num_cores_all_to_all_first_stage;
    }
    return width_index < ctx.workers.num_cores_all_to_all;
}

namespace {

// The multicast range this sender covers, plus its own position within the grid.
std::vector<uint32_t> reader_sender_named_values(
    const CoreCoord& core, const RuntimeArgsContext& ctx, IDevice* device) {
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

    uint32_t start_x = 0;
    uint32_t start_y = 0;
    if (ctx.grid.mcast_1d) {
        start_x = core.x - ctx.core_ranges.start_core.x;
        start_y = core.y - ctx.core_ranges.start_core.y;
    } else if (ctx.grid.row_wise) {
        start_x = core.x - ctx.core_ranges.start_core.x;
    } else {
        start_y = core.y - ctx.core_ranges.start_core.y;
    }
    return {mcast_start.x, mcast_start.y, mcast_end.x, mcast_end.y, start_x, start_y};
}

// The coordinate block an all-to-all worker walks to reach its remote peers: every X coordinate of
// the multicast grid followed by every Y coordinate. In the 2D cases only one of the two axes varies,
// so the other contributes the single coordinate of this core's own row or column.
std::vector<uint32_t> gather_coord_varargs(const CoreIndices& idx, const RuntimeArgsContext& ctx) {
    std::vector<uint32_t> varargs;
    varargs.reserve(ctx.mcast_noc_x.size() + ctx.mcast_noc_y.size());
    if (ctx.grid.mcast_1d) {
        varargs.insert(varargs.end(), ctx.mcast_noc_x.begin(), ctx.mcast_noc_x.end());
        varargs.insert(varargs.end(), ctx.mcast_noc_y.begin(), ctx.mcast_noc_y.end());
    } else if (ctx.grid.row_wise) {
        varargs.insert(varargs.end(), ctx.mcast_noc_x.begin(), ctx.mcast_noc_x.end());
        varargs.push_back(ctx.mcast_noc_y[idx.height_index]);
    } else {
        varargs.push_back(ctx.mcast_noc_x[idx.height_index]);
        varargs.insert(varargs.end(), ctx.mcast_noc_y.begin(), ctx.mcast_noc_y.end());
    }
    return varargs;
}

// A core that only waits for the multicast needs one coordinate pair: the sender's.
std::vector<uint32_t> sender_coord_varargs(const CoreIndices& idx, const RuntimeArgsContext& ctx) {
    if (ctx.grid.mcast_1d) {
        return {ctx.mcast_noc_x[0], ctx.mcast_noc_y[0]};
    }
    if (ctx.grid.row_wise) {
        return {ctx.mcast_noc_x[0], ctx.mcast_noc_y[idx.height_index]};
    }
    return {ctx.mcast_noc_x[idx.height_index], ctx.mcast_noc_y[0]};
}

// Which storage-core segments this worker's output block spans, three values each: the byte count and
// the destination node's coordinates. Advances the shared storage-core cursor, so it must be called
// once per core in shard order.
std::vector<uint32_t> write_back_varargs(
    const RuntimeArgsContext& ctx,
    uint32_t& current_storage_core,
    uint32_t& current_storage_core_offset,
    uint32_t& num_segments_out,
    uint32_t& storage_core_start_offset_out) {
    std::vector<uint32_t> args;
    storage_core_start_offset_out = current_storage_core_offset * ctx.out_single_tile_size;
    num_segments_out = 0;
    uint32_t worker_offset = 0;

    while (worker_offset < ctx.block_wt) {
        uint32_t tiles_available = ctx.block_wt_resharded - current_storage_core_offset;
        uint32_t tiles_left = ctx.block_wt - worker_offset;
        uint32_t tiles_to_write = std::min(tiles_left, tiles_available);

        num_segments_out += 1;
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
    return args;
}

m2::KernelRunArgs* find_run_args(m2::ProgramRunArgs& run_args, const m2::KernelSpecName& kernel) {
    for (auto& entry : run_args.kernel_run_args) {
        if (entry.kernel == kernel) {
            return &entry;
        }
    }
    return nullptr;
}

}  // namespace

RunArgsAndWriterVarargs build_run_args(
    const std::vector<CoreCoord>& cores,
    const RuntimeArgsContext& ctx,
    const SpecConfig& config,
    IDevice* device,
    const Tensor& input,
    const std::optional<Tensor>& residual,
    const std::optional<Tensor>& gamma,
    const std::optional<Tensor>& beta,
    const std::optional<Tensor>& stats,
    const std::optional<Tensor>& recip,
    const Tensor& output) {
    // The same three predicates that decide which kernels the spec declares.
    const bool has_reader_receiver_all_to_all =
        ctx.grid.use_mcast && !ctx.core_ranges.all_to_all_workers_except_sender.empty();
    const bool has_not_all_to_all_workers = ctx.workers.num_none_all_to_all_workers > 0;

    m2::ProgramRunArgs run_args;
    run_args.kernel_run_args.push_back(m2::KernelRunArgs{.kernel = READER_SENDER});
    if (has_reader_receiver_all_to_all) {
        run_args.kernel_run_args.push_back(m2::KernelRunArgs{.kernel = READER_RECEIVER_ALL_TO_ALL});
    }
    if (has_not_all_to_all_workers) {
        run_args.kernel_run_args.push_back(m2::KernelRunArgs{.kernel = READER_RECEIVER});
    }
    run_args.kernel_run_args.push_back(m2::KernelRunArgs{.kernel = WRITER_SENDER});
    if (has_not_all_to_all_workers) {
        run_args.kernel_run_args.push_back(m2::KernelRunArgs{.kernel = WRITER_RECEIVER});
    }
    run_args.kernel_run_args.push_back(m2::KernelRunArgs{.kernel = COMPUTE_ALL_TO_ALL});
    if (has_not_all_to_all_workers) {
        run_args.kernel_run_args.push_back(m2::KernelRunArgs{.kernel = COMPUTE_NOT_ALL_TO_ALL});
    }

    auto& reader_sender = *find_run_args(run_args, READER_SENDER);
    auto& writer_sender = *find_run_args(run_args, WRITER_SENDER);
    auto& compute_all_to_all = *find_run_args(run_args, COMPUTE_ALL_TO_ALL);
    // Absent when this configuration places no such kernel; the loop below only reaches them for
    // cores that fall in the matching group, which is empty in that case.
    auto* reader_receiver_all_to_all = find_run_args(run_args, READER_RECEIVER_ALL_TO_ALL);
    auto* reader_receiver = find_run_args(run_args, READER_RECEIVER);
    auto* writer_receiver = find_run_args(run_args, WRITER_RECEIVER);
    auto* compute_not_all_to_all = find_run_args(run_args, COMPUTE_NOT_ALL_TO_ALL);

    uint32_t current_storage_core = 0;
    uint32_t current_storage_core_offset = 0;

    for (uint32_t i = 0; i < cores.size(); ++i) {
        const auto& core = cores[i];
        const auto idx = CoreIndices::compute(i, core, ctx);
        const bool is_all_to_all = idx.is_all_to_all(ctx);

        //------------------------------------------------------------------
        // Compute
        //------------------------------------------------------------------
        auto& compute = is_all_to_all ? compute_all_to_all : *compute_not_all_to_all;
        m2::AddRuntimeArgsForNode(
            compute.runtime_arg_values, core, {{"num_reduce_tiles_per_block_h", idx.num_reduce_tiles_per_block_h}});
        if (is_all_to_all) {
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
            const bool is_second_stage_reader =
                ctx.grid.use_two_stage_reduce && idx.width_index < ctx.workers.num_cores_all_to_all_first_stage;
            m2::AddRuntimeArgsForNode(
                compute.runtime_arg_values,
                core,
                {{"num_rows_per_all_to_all_worker", num_rows},
                 {"use_two_stage_reduce", static_cast<uint32_t>(ctx.grid.use_two_stage_reduce)},
                 {"is_second_stage_reader", static_cast<uint32_t>(is_second_stage_reader)}});
            if (ctx.is_post_all_gather) {
                m2::AddRuntimeArgsForNode(
                    compute.runtime_arg_values, core, {{"num_distributed_blocks", ctx.num_distributed_devices}});
            }
        }
        if (config.use_welford) {
            m2::AddRuntimeArgsForNode(compute.runtime_arg_values, core, {{"welford_reduce_w", idx.welford_reduce_w}});
            if (is_all_to_all) {
                // The global width-block index of the last real (partial) block, and this core's own
                // width-block index. The Welford cross-core combine uses these to weight each combined
                // block or row by its true logical width: the partial block sits at a single global
                // position, not in every row.
                m2::AddRuntimeArgsForNode(
                    compute.runtime_arg_values,
                    core,
                    {{"boundary_width_index", ctx.last_core_width_index}, {"my_width_index", idx.width_index}});
            }
        }

        //------------------------------------------------------------------
        // Reader
        //------------------------------------------------------------------
        if (idx.width_index == 0) {
            const auto named = reader_sender_named_values(core, ctx, device);
            m2::AddRuntimeArgsForNode(
                reader_sender.runtime_arg_values,
                core,
                {{"mcast_dest_noc_start_x", named[0]},
                 {"mcast_dest_noc_start_y", named[1]},
                 {"mcast_dest_noc_end_x", named[2]},
                 {"mcast_dest_noc_end_y", named[3]},
                 {"start_x", named[4]},
                 {"start_y", named[5]}});
            reader_sender.advanced_options.runtime_varargs[core] = gather_coord_varargs(idx, ctx);
        } else if (is_all_to_all) {
            const bool is_last_all_to_all_worker =
                ctx.grid.use_two_stage_reduce
                    ? idx.width_index_two_stage == ctx.workers.num_cores_all_to_all_first_stage - 1
                    : idx.width_index == ctx.workers.num_cores_all_to_all - 1;
            const bool is_second_stage_reader =
                ctx.grid.use_two_stage_reduce && idx.width_index < ctx.workers.num_cores_all_to_all_first_stage;
            uint32_t start_x = 0;
            uint32_t start_y = 0;
            if (ctx.grid.mcast_1d) {
                start_x = core.x - ctx.core_ranges.start_core.x;
                start_y = core.y - ctx.core_ranges.start_core.y;
            } else if (ctx.grid.row_wise) {
                start_x = core.x - ctx.core_ranges.start_core.x;
            } else {
                start_y = core.y - ctx.core_ranges.start_core.y;
            }
            m2::AddRuntimeArgsForNode(
                reader_receiver_all_to_all->runtime_arg_values,
                core,
                {{"is_last_all_to_all_worker", static_cast<uint32_t>(is_last_all_to_all_worker)},
                 {"all_to_all_tile_offset_bytes", idx.all_to_all_worker_tile_offset_bytes},
                 {"is_second_stage_reader", static_cast<uint32_t>(is_second_stage_reader)},
                 {"start_x", start_x},
                 {"start_y", start_y}});
            reader_receiver_all_to_all->advanced_options.runtime_varargs[core] = gather_coord_varargs(idx, ctx);
        } else {
            m2::AddRuntimeArgsForNode(
                reader_receiver->runtime_arg_values,
                core,
                {{"is_last_all_to_all_worker", 0},
                 {"all_to_all_tile_offset_bytes", idx.all_to_all_worker_tile_offset_bytes},
                 {"is_second_stage_reader", 0},
                 {"start_x", 0},
                 {"start_y", 0}});
            reader_receiver->advanced_options.runtime_varargs[core] = sender_coord_varargs(idx, ctx);
        }

        //------------------------------------------------------------------
        // Writer
        //------------------------------------------------------------------
        auto& writer = is_all_to_all ? writer_sender : *writer_receiver;
        if (!config.use_welford) {
            // A two-stage reduce's second-stage cores have already had the cross-core average applied
            // by the first stage, so they must not apply it again.
            const uint32_t packed_cinv = (is_all_to_all && ctx.grid.use_two_stage_reduce &&
                                          idx.width_index >= ctx.workers.num_cores_all_to_all_first_stage)
                                             ? ctx.packed_cinv_value_one
                                             : ctx.packed_cinv_value;
            m2::AddRuntimeArgsForNode(
                writer.runtime_arg_values, core, {{"scalar_c", packed_cinv}, {"scalar_w", ctx.packed_winv_value}});
            if (!config.is_pre_all_gather) {
                m2::AddRuntimeArgsForNode(writer.runtime_arg_values, core, {{"eps", ctx.eps_u}});
            }
        }
        if (!config.is_pre_all_gather || config.do_col_mask) {
            m2::AddRuntimeArgsForNode(
                writer.runtime_arg_values, core, {{"width_shard_tile_start_id", idx.width_shard_tile_start_id}});
        }
        if (ctx.writes_back) {
            uint32_t num_segments = 0;
            uint32_t storage_core_start_offset = 0;
            auto segments = write_back_varargs(
                ctx, current_storage_core, current_storage_core_offset, num_segments, storage_core_start_offset);
            m2::AddRuntimeArgsForNode(
                writer.runtime_arg_values,
                core,
                {{"num_segments_to_write_back", num_segments},
                 {"storage_core_start_offset", storage_core_start_offset}});
            writer.advanced_options.runtime_varargs[core] = std::move(segments);
        }
    }

    // The write-back segment block is ragged: how many storage cores a worker's output block spans
    // depends on where the shared storage cursor happened to be. The vararg count is a per-kernel
    // property, so every node declares the longest block and the shorter ones are zero-padded. The
    // kernel reads exactly num_segments_to_write_back segments, so the padding is never looked at.
    uint32_t writer_num_varargs = 0;
    if (ctx.writes_back) {
        size_t longest = 0;
        for (auto* writer : {&writer_sender, writer_receiver}) {
            if (writer == nullptr) {
                continue;
            }
            for (const auto& [node, varargs] : writer->advanced_options.runtime_varargs) {
                longest = std::max(longest, varargs.size());
            }
        }
        for (auto* writer : {&writer_sender, writer_receiver}) {
            if (writer == nullptr) {
                continue;
            }
            for (auto& [node, varargs] : writer->advanced_options.runtime_varargs) {
                varargs.resize(longest, 0);
            }
        }
        writer_num_varargs = static_cast<uint32_t>(longest);
    }

    //----------------------------------------------------------------------
    // Tensor arguments
    //----------------------------------------------------------------------
    run_args.tensor_args.emplace(INPUT, input.mesh_tensor());
    run_args.tensor_args.emplace(OUTPUT, output.mesh_tensor());
    if (config.has_b && !config.is_post_all_gather) {
        run_args.tensor_args.emplace(RESIDUAL, residual.value().mesh_tensor());
    }
    if (config.has_gamma && !config.is_pre_all_gather) {
        run_args.tensor_args.emplace(GAMMA_T, gamma.value().mesh_tensor());
    }
    if (config.has_beta && !config.is_pre_all_gather) {
        run_args.tensor_args.emplace(BETA_T, beta.value().mesh_tensor());
    }
    if (config.is_post_all_gather) {
        run_args.tensor_args.emplace(STATS_T, stats.value().mesh_tensor());
    }
    if (config.use_welford) {
        run_args.tensor_args.emplace(RECIP, recip.value().mesh_tensor());
    }

    return RunArgsAndWriterVarargs{.run_args = std::move(run_args), .writer_num_varargs = writer_num_varargs};
}

}  // namespace ttnn::prim::sharded_layernorm_helpers

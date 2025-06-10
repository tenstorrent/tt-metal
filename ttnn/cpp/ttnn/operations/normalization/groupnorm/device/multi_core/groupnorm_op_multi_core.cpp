// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/circular_buffer_config.hpp>
#include "ttnn/operations/normalization/groupnorm/device/groupnorm_op.hpp"
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/math.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>

#include <optional>

using uint32_t = std::uint32_t;
using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::operations::normalization {

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {
inline bool is_dram(const Tensor& input_tensor) {
    return input_tensor.memory_config().buffer_type() == BufferType::DRAM;
}
inline bool is_dram(const std::optional<const Tensor>& input_tensor) {
    return input_tensor.has_value() ? is_dram(input_tensor.value()) : true;
}
inline bool is_dram(const Buffer* b) { return b->buffer_type() == BufferType::DRAM; }

inline bool cbs_fit_in_DRAM(
    uint32_t in0_CB_size,
    uint32_t in_CB_size,
    uint32_t in2_CB_size,
    uint32_t in3_CB_size,
    uint32_t in5_CB_size,
    uint32_t in6_CB_size,
    uint32_t in_mask_CB_size,
    uint32_t repack_CB_size,
    uint32_t x_CB_size,
    uint32_t xmm_CB_size,
    uint32_t ex_partial_CB_size,
    uint32_t ex_global_CB_size,
    uint32_t ex2_global_CB_size,
    uint32_t xmm2_CB_size,
    uint32_t xmm3_CB_size,
    uint32_t ex2pe_CB_size,
    uint32_t out_CB_size,
    uint32_t l1_size) {
    uint32_t sum = 0;
    sum += in0_CB_size;
    sum += in_CB_size;
    sum += in2_CB_size;
    sum += in3_CB_size;
    sum += in5_CB_size;
    sum += in6_CB_size;
    sum += in_mask_CB_size;
    sum += repack_CB_size;
    sum += x_CB_size;
    sum += xmm_CB_size;
    sum += ex_partial_CB_size;
    sum += ex_global_CB_size;
    sum += ex2_global_CB_size;
    sum += xmm2_CB_size;
    sum += xmm3_CB_size;
    sum += ex2pe_CB_size;
    sum += out_CB_size;
    return sum < l1_size;
}

int get_max_subblock(uint32_t n, uint32_t max_subblock_w) {
    if (n <= max_subblock_w) {
        return n;
    }

    for (int quotient = max_subblock_w; quotient > 1; --quotient) {
        if (n % quotient == 0) {
            return quotient;
        }
    }
    return 1;
}
bool is_rectangle_grid(const std::vector<CoreCoord>& core_coords) {
    if (core_coords.empty()) {
        return true;
    }

    int min_x = std::numeric_limits<int>::max();
    int max_x = std::numeric_limits<int>::min();
    int min_y = std::numeric_limits<int>::max();
    int max_y = std::numeric_limits<int>::min();

    for (const auto& coord : core_coords) {
        min_x = std::min(min_x, static_cast<int>(coord.x));
        max_x = std::max(max_x, static_cast<int>(coord.x));
        min_y = std::min(min_y, static_cast<int>(coord.y));
        max_y = std::max(max_y, static_cast<int>(coord.y));
    }

    return ((max_x - min_x + 1) * (max_y - min_y + 1)) == core_coords.size();
}
void split_and_form_rectangle_grids(
    std::vector<CoreCoord>& group,
    std::vector<CoreCoord>& mcast_group_first,
    std::vector<CoreCoord>& mcast_group_mid,
    std::vector<CoreCoord>& mcast_group_last) {
    int remove_front = 0;
    int remove_back = 0;
    int min_total_removal = group.size();

    for (int front = 0; front <= group.size(); ++front) {
        for (int back = 0; front + back <= group.size(); ++back) {
            if (is_rectangle_grid(std::vector<CoreCoord>(group.begin() + front, group.end() - back))) {
                int total_removal = front + back;
                if (total_removal < min_total_removal) {
                    min_total_removal = total_removal;
                    remove_front = front;
                    remove_back = back;
                }
            }
        }
    }

    // Pop and push the front outliers
    for (int i = 0; i < remove_front; ++i) {
        mcast_group_first.push_back(mcast_group_mid.front());
        mcast_group_mid.erase(mcast_group_mid.begin());
    }

    // Pop and push the back outliers
    for (int i = 0; i < remove_back; ++i) {
        mcast_group_last.push_back(mcast_group_mid.back());
        mcast_group_mid.pop_back();
    }
}

std::pair<uint32_t, uint32_t> find_max_tile_span(uint32_t W, uint32_t group_size) {
    uint32_t current_position = 0;
    uint32_t max_tile_span = 0;
    uint32_t num_groups_before_start_again_at_tile_beginning = -1;
    bool calc_num_groups_before_start_again_at_tile_beginning = true;

    while (current_position < W) {
        uint32_t group_end = current_position + group_size;
        uint32_t start_tile = current_position / TILE_WIDTH;
        uint32_t end_tile = (group_end - 1) / TILE_WIDTH;
        uint32_t current_tile_span = end_tile - start_tile + 1;

        max_tile_span = std::max(max_tile_span, current_tile_span);

        current_position = group_end;

        if (current_position % TILE_WIDTH == 0 and calc_num_groups_before_start_again_at_tile_beginning) {
            num_groups_before_start_again_at_tile_beginning = current_position / group_size;
            calc_num_groups_before_start_again_at_tile_beginning = false;
        }
    }

    return {max_tile_span, num_groups_before_start_again_at_tile_beginning};
}
}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

operation::ProgramWithCallbacks groupnorm_multi_core_sharded(
    const Tensor& a,
    const std::optional<const Tensor>& gamma,
    const std::optional<const Tensor>& beta,
    const std::optional<const Tensor>& input_mask,
    Tensor& output,
    float eps,
    const uint32_t num_groups,
    const uint32_t num_batches,
    MathFidelity fidelity,
    DataType im_data_format,
    CoreCoord grid_size,
    bool inplace) {
    using namespace CMAKE_UNIQUE_NAMESPACE;
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
    uint32_t datum_size_bytes = 2;  // bfloat16

    TT_FATAL(
        out_data_format == in_data_format,
        "Input and output must have the same data format, but input has {} and output has {}",
        in_data_format,
        out_data_format);

    // tile sizes
    uint32_t in_single_tile_size = tt::tt_metal::detail::TileSize(in_data_format);
    uint32_t single_tile_size = tt::tt_metal::detail::TileSize(cb_data_format);
    uint32_t out_single_tile_size = tt::tt_metal::detail::TileSize(out_data_format);
    uint32_t gamma_beta_single_tile_size = tt::tt_metal::detail::TileSize(gamma_beta_cb_data_format);
    uint32_t in_mask_single_tile_size = tt::tt_metal::detail::TileSize(in_mask_cb_data_format);
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
    const auto shape = a.padded_shape();
    uint32_t H = shape[2] * num_batches;
    uint32_t Ht = H / TILE_HEIGHT;
    uint32_t W = shape[3];
    uint32_t Wt = W / TILE_WIDTH;
    uint32_t num_datum_row_per_group = W / num_groups;
    uint32_t num_datum_row_per_group_mod_tile_w =
        num_datum_row_per_group % TILE_WIDTH == 0 ? TILE_WIDTH : num_datum_row_per_group % TILE_WIDTH;
    uint32_t group_size = W / num_groups;
    // grid
    uint32_t num_cores_c = grid_size.x;
    uint32_t num_cores_r = grid_size.y;
    // uint32_t num_cores = num_cores_c * num_cores_r;
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
    if (shard_orientation == ShardOrientation::ROW_MAJOR and num_groups_per_core == 1) {
        TT_FATAL(
            num_cores_c % num_groups == 0,
            "for RM shard, when each group is split across cores, num_cores_c ({}) must be divisible by num_groups "
            "({})",
            num_cores_c,
            num_groups);
    } else if (shard_orientation == ShardOrientation::COL_MAJOR and num_groups_per_core == 1) {
        TT_FATAL(
            num_cores_r % num_groups == 0,
            "for CM shard, when each group is split across cores, num_cores_r ({}) must be divisible by num_groups "
            "({})",
            num_cores_r,
            num_groups);
    }

    if (per_core_N != W) {  // block sharded
        if (shard_orientation == ShardOrientation::ROW_MAJOR and num_batches_per_core == 1) {
            TT_FATAL(
                num_cores_r % num_batches == 0,
                "for RM shard, when each batch is split across cores, num_cores_r ({}) must be divisible by "
                "num_batches ({})",
                num_cores_r,
                num_batches);
        } else if (shard_orientation == ShardOrientation::COL_MAJOR and num_groups_per_core == 1) {
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
    auto in0_addr = a.buffer()->address();
    auto out_addr = output.buffer()->address();
    // gamma, beta addr
    auto gamma_dram_addr = gamma.has_value() ? gamma.value().buffer()->address() : 0;
    auto beta_dram_addr = beta.has_value() ? beta.value().buffer()->address() : 0;
    auto input_mask_dram_addr = input_mask.has_value() ? input_mask.value().buffer()->address() : 0;
    // num tiles for a, gamma, beta
    uint32_t num_tiles = a.physical_volume() / TILE_HW;
    uint32_t num_gamma_tiles = gamma.has_value() ? gamma.value().physical_volume() / TILE_HW : 0;
    uint32_t num_beta_tiles = beta.has_value() ? beta.value().physical_volume() / TILE_HW : 0;
    uint32_t num_input_mask_tiles = input_mask.has_value() ? input_mask.value().physical_volume() / TILE_HW : 0;

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
    uint32_t in2_CB_size = single_tile_size;
    // in3 - eps
    uint32_t in3_CB_size = single_tile_size;
    // gamma
    // uint32_t gamma_beta_num_cols_tile_per_core = block_wt * num_groups_per_core;
    uint32_t gamma_beta_num_cols_tile_per_core = per_core_Nt;
    uint32_t in5_CB_size = gamma_beta_num_cols_tile_per_core * gamma_beta_single_tile_size;
    // beta
    uint32_t in6_CB_size = gamma_beta_num_cols_tile_per_core * gamma_beta_single_tile_size;
    // input mask
    uint32_t input_mask_num_tiles_per_core = block_wt * num_groups_per_core;
    uint32_t in_mask_CB_size = block_wt * in_mask_single_tile_size * 2;  // double buffer
    // repack cb
    uint32_t repack_CB_size = per_core_Nt * in_single_tile_size * 2;  // double buffer
    // itermediate buffers
    uint32_t interm_block_tiles = block_ht * block_wt;
    uint32_t x_CB_size = interm_block_tiles * single_tile_size;
    uint32_t xmm_CB_size = interm_block_tiles * single_tile_size;
    uint32_t ex_partial_CB_size = single_tile_size;   // partial Ex
    uint32_t ex_global_CB_size = ex_partial_CB_size;  // the final result Ex
    uint32_t xmm2_CB_size = interm_block_tiles * single_tile_size;
    uint32_t ex2pe_CB_size = ex_partial_CB_size;
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
    bool use_mcast = num_cores_per_batch > 1 or num_cores_per_group > 1;
    uint32_t start_core_x = 0;
    uint32_t start_core_y = 0;

    // create a vector of cores, in either RM or CM
    std::vector<CoreCoord> core_coords =
        grid_to_cores(num_cores, num_cores_c, num_cores_r, shard_orientation == ShardOrientation::ROW_MAJOR);
    for (int i = 0; i < core_coords.size(); ++i) {
        log_debug(tt::LogOp, "worker coord: {} {}", core_coords[i].x, core_coords[i].y);
    }
    std::vector<std::vector<CoreCoord>> core_coords2D;
    if (shard_orientation == ShardOrientation::ROW_MAJOR) {
        for (int i = 0; i < num_cores_c / num_cores_per_group; ++i) {
            for (int j = 0; j < num_cores_r; ++j) {
                std::vector<CoreCoord> temp;
                for (int k = 0; k < num_cores_per_group; ++k) {
                    temp.push_back(CoreCoord{(std::size_t)(k + i * num_cores_per_group), (std::size_t)j});
                }
                core_coords2D.push_back(temp);
            }
        }
    } else {
        for (int i = 0; i < num_cores_r / num_cores_per_group; ++i) {
            for (int j = 0; j < num_cores_c; ++j) {
                std::vector<CoreCoord> temp;
                for (int k = 0; k < num_cores_per_group; ++k) {
                    temp.push_back(CoreCoord{(std::size_t)j, (std::size_t)(k + i * num_cores_per_group)});
                }
                core_coords2D.push_back(temp);
            }
        }
    }

    // one mcast core per batch per group
    std::set<CoreRange> mcast_sender_core_ranges;
    std::set<CoreRange> mcast_receiver_core_ranges;
    uint32_t core_index = 0;
    uint32_t core_index_offset = 0;
    for (int i = 0; i < num_batches / num_batches_per_core; ++i) {
        uint32_t core_index = core_index_offset;
        for (int j = 0; j < num_groups / num_groups_per_core; ++j) {
            mcast_sender_core_ranges.insert(CoreRange(core_coords[core_index]));
            core_index += num_cores_per_group;
            core_index_offset += num_cores_per_batch * num_cores_per_group;
        }
    }
    for (auto& coord : mcast_sender_core_ranges) {
        log_debug(tt::LogOp, "mcast sender coord: {} {}", coord.start_coord.x, coord.start_coord.y);
    }
    for (int i = 0; i < num_cores; ++i) {
        // not found in mcast sender
        if (mcast_sender_core_ranges.find(CoreRange(core_coords[i])) == mcast_sender_core_ranges.end()) {
            mcast_receiver_core_ranges.insert(CoreRange(core_coords[i]));
        }
    }
    for (auto& coord : mcast_receiver_core_ranges) {
        log_debug(tt::LogOp, "mcast receiver coord: {} {}", coord.start_coord.x, coord.start_coord.y);
    }
    CoreRangeSet mcast_sender_cores = CoreRangeSet(mcast_sender_core_ranges);
    CoreRangeSet mcast_receiver_cores = CoreRangeSet(mcast_receiver_core_ranges);
    // mcast groups
    std::vector<std::vector<CoreCoord>> mcast_groups;
    int group_index = -1;
    if (is_height_sharding) {
        for (int i = 0; i < num_cores; ++i) {
            if (mcast_sender_core_ranges.find(CoreRange(core_coords[i])) != mcast_sender_core_ranges.end()) {
                group_index += 1;
            }
            if (group_index >= mcast_groups.size()) {
                mcast_groups.push_back(std::vector<CoreCoord>());  // Add a new group
            }
            mcast_groups[group_index].push_back(core_coords[i]);
        }
    } else {
        for (int i = 0; i < core_coords2D.size(); ++i) {
            for (int j = 0; j < core_coords2D[i].size(); ++j) {
                if (mcast_sender_core_ranges.find(CoreRange(core_coords2D[i][j])) != mcast_sender_core_ranges.end()) {
                    group_index += 1;
                }
                if (group_index >= mcast_groups.size()) {
                    mcast_groups.push_back(std::vector<CoreCoord>());  // Add a new group
                }
                mcast_groups[group_index].push_back(core_coords2D[i][j]);
            }
        }
    }
    for (int i = 0; i < mcast_groups.size(); ++i) {
        for (int j = 0; j < mcast_groups[i].size(); ++j) {
            log_debug(tt::LogOp, "mcast group: {} coord: {} {}", i, mcast_groups[i][j].x, mcast_groups[i][j].y);
        }
    }
    // how many cores in a mcast group
    uint32_t num_cores_per_mcast_group = mcast_groups[0].size();
    // Mcast args
    auto reduce_sender_semaphore_id = tt::tt_metal::CreateSemaphore(program, all_cores, INVALID);
    auto reduce_receiver_semaphore_id = tt::tt_metal::CreateSemaphore(program, all_cores, INVALID);
    // reader defines
    std::map<string, string> reader_mcast_sender_defines;
    std::map<string, string> reader_mcast_receiver_defines;
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
        (std::uint32_t)num_groups_per_core * num_batches_per_core,
        (std::uint32_t)per_core_Nt,
        (std::uint32_t)per_core_N_bytes_padded,
        (std::uint32_t)per_core_Nt * TILE_WIDTH * datum_size_bytes,
        (std::uint32_t)datum_size_bytes,
        (std::uint32_t)per_core_Mt,
        (std::uint32_t)TILE_HEIGHT};
    std::vector<uint32_t> reader_mcast_receiver_compile_time_args = {
        (std::uint32_t)reduce_receiver_semaphore_id,
        (std::uint32_t)reduce_sender_semaphore_id,
        (std::uint32_t)num_groups_per_core * num_batches_per_core,
        (std::uint32_t)per_core_Nt,
        (std::uint32_t)per_core_N_bytes_padded,
        (std::uint32_t)per_core_Nt * TILE_WIDTH * datum_size_bytes,
        (std::uint32_t)per_core_Mt,
        (std::uint32_t)TILE_HEIGHT};
    tt::tt_metal::NOC reader_noc = tt::tt_metal::detail::GetPreferredNOCForDRAMWrite(device->arch());
    tt::tt_metal::NOC writer_noc = tt::tt_metal::detail::GetPreferredNOCForDRAMRead(device->arch());
    // reader kernel
    auto reader_mcast_sender_kernels_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/dataflow/"
        "reader_mcast_sender_unary_sharded_gn_v2.cpp",
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
            "ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/dataflow/"
            "reader_mcast_receiver_unary_sharded_gn_v2.cpp",
            mcast_receiver_cores,
            tt::tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                .noc = reader_noc,
                .compile_args = reader_mcast_receiver_compile_time_args,
                .defines = reader_mcast_receiver_defines});
    }

    // writer defines
    std::map<string, string> writer_defines;
    // writer compile time args
    std::vector<uint32_t> writer_mcast_sender_compile_time_args = {
        1,
        (std::uint32_t)gamma.has_value(),
        (std::uint32_t)beta.has_value(),
        (std::uint32_t)is_dram(gamma),
        (std::uint32_t)is_dram(beta),
        (std::uint32_t)is_dram(input_mask),
        (std::uint32_t)gamma_beta_num_cols_tile_per_core,
        (std::uint32_t)per_core_N,
        (std::uint32_t)per_core_N * datum_size_bytes,
        (std::uint32_t)per_core_Nt * TILE_WIDTH * datum_size_bytes,
        (std::uint32_t)num_groups_per_core,
        (std::uint32_t)num_batches_per_core,
        (std::uint32_t)block_wt};

    if (gamma.has_value() and gamma.value().layout() == Layout::ROW_MAJOR) {
        auto gamma_stick_size = gamma.value().padded_shape()[3] * gamma.value().element_size();
        bool gamma_stick_size_is_power_of_two = is_power_of_two_at_least_32(gamma_stick_size);
        writer_mcast_sender_compile_time_args.push_back((std::uint32_t)gamma_stick_size_is_power_of_two);
        if (gamma_stick_size_is_power_of_two) {
            uint32_t gamma_log2_stick_size =
                gamma_stick_size_is_power_of_two ? (std::uint32_t)std::log2(gamma_stick_size) : 0;
            writer_mcast_sender_compile_time_args.push_back((std::uint32_t)gamma_log2_stick_size);
        } else {
            writer_mcast_sender_compile_time_args.push_back(gamma_stick_size);
        }
    } else if (beta.has_value() and beta.value().layout() == Layout::ROW_MAJOR) {
        auto beta_stick_size = beta.value().padded_shape()[3] * beta.value().element_size();
        bool beta_stick_size_is_power_of_two = is_power_of_two_at_least_32(beta_stick_size);
        writer_mcast_sender_compile_time_args.push_back((std::uint32_t)beta_stick_size_is_power_of_two);
        if (beta_stick_size_is_power_of_two) {
            uint32_t beta_log2_stick_size =
                beta_stick_size_is_power_of_two ? (std::uint32_t)std::log2(beta_stick_size) : 0;
            writer_mcast_sender_compile_time_args.push_back((std::uint32_t)beta_log2_stick_size);
        } else {
            writer_mcast_sender_compile_time_args.push_back(beta_stick_size);
        }
    } else {
        writer_mcast_sender_compile_time_args.push_back(0);
        writer_mcast_sender_compile_time_args.push_back(0);
    }

    // writer kernel
    bool use_row_major_kernel = true;
    std::string writer_kernel =
        "ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/dataflow/writer_unary_sharded_gn_rm_gb_v2.cpp";
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
    std::map<string, string> eltwise_binary_defines;
    if (reader_repack_output) {
        eltwise_binary_defines["READER_REPACK"] = "1";
    }
    if (tilize_in) {
        eltwise_binary_defines["TILIZE_IN"] = "1";
    }
    if (untilize_out) {
        eltwise_binary_defines["UNTILIZE_OUT"] = "1";
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
        (std::uint32_t)num_datum_row_per_group - (block_wt - 1) * TILE_WIDTH

    };
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
        (std::uint32_t)num_datum_row_per_group - (block_wt - 1) * TILE_WIDTH};
    // compute kernel
    bool fp32_dest_acc_en = false;
    bool math_approx_mode = true;
    auto mcast_sender_compute_kernels_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/compute/groupnorm_sharded_v2.cpp",
        mcast_sender_cores,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .math_approx_mode = math_approx_mode,
            .compile_args = mcast_sender_compute_compile_time_args,
            .defines = eltwise_binary_defines});
    auto mcast_receiver_compute_kernels_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/compute/groupnorm_sharded_v2.cpp",
        mcast_receiver_cores,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = fidelity,
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

    // in - stores tilized input
    uint32_t in_cb_index = tt::CBIndex::c_1;
    tt::tt_metal::CircularBufferConfig in_cb_config =
        tt::tt_metal::CircularBufferConfig(in_CB_size, {{in_cb_index, in_data_format}})
            .set_page_size(in_cb_index, in_single_tile_size);
    auto cb_in = tt::tt_metal::CreateCircularBuffer(program, all_cores, in_cb_config);
    // out - stores tilized output
    if (untilize_out) {
        uint32_t out_cb_index = tt::CBIndex::c_30;
        tt::tt_metal::CircularBufferConfig out_cb_config =
            tt::tt_metal::CircularBufferConfig(in_CB_size, {{out_cb_index, in_data_format}})
                .set_page_size(out_cb_index, in_single_tile_size);
        auto cb_out = tt::tt_metal::CreateCircularBuffer(program, all_cores, out_cb_config);
    }
    // in2 scaler - for partial Ex
    uint32_t in2_cb_index = tt::CBIndex::c_2;
    tt::tt_metal::CircularBufferConfig in2_cb_config =
        tt::tt_metal::CircularBufferConfig(in2_CB_size, {{in2_cb_index, cb_data_format}})
            .set_page_size(in2_cb_index, single_tile_size);
    auto cb_in2 = tt::tt_metal::CreateCircularBuffer(program, all_cores, in2_cb_config);
    // in3 eps
    uint32_t in3_cb_index = tt::CBIndex::c_3;
    tt::tt_metal::CircularBufferConfig in3_cb_config =
        tt::tt_metal::CircularBufferConfig(in3_CB_size, {{in3_cb_index, cb_data_format}})
            .set_page_size(in3_cb_index, single_tile_size);
    auto cb_in3 = tt::tt_metal::CreateCircularBuffer(program, all_cores, in3_cb_config);
    // in4 scaler-c
    uint32_t in4_cb_index = tt::CBIndex::c_4;
    tt::tt_metal::CircularBufferConfig in4_cb_config =
        tt::tt_metal::CircularBufferConfig(in2_CB_size, {{in4_cb_index, cb_data_format}})
            .set_page_size(in4_cb_index, single_tile_size);
    auto cb_in4 = tt::tt_metal::CreateCircularBuffer(program, all_cores, in4_cb_config);
    // gamma
    if (gamma.has_value()) {
        uint32_t in5_cb_index = tt::CBIndex::c_5;
        tt::tt_metal::CircularBufferConfig in5_cb_config =
            tt::tt_metal::CircularBufferConfig(in5_CB_size, {{in5_cb_index, gamma_beta_cb_data_format}})
                .set_page_size(in5_cb_index, gamma_beta_single_tile_size);
        auto cb_in5 = tt::tt_metal::CreateCircularBuffer(program, all_cores, in5_cb_config);
    }
    // beta
    if (beta.has_value()) {
        uint32_t in6_cb_index = tt::CBIndex::c_6;
        tt::tt_metal::CircularBufferConfig in6_cb_config =
            tt::tt_metal::CircularBufferConfig(in6_CB_size, {{in6_cb_index, gamma_beta_cb_data_format}})
                .set_page_size(in6_cb_index, gamma_beta_single_tile_size);
        auto cb_in6 = tt::tt_metal::CreateCircularBuffer(program, all_cores, in6_cb_config);
    }
    // input mask
    if (input_mask.has_value()) {
        uint32_t in_mask_cb_index = tt::CBIndex::c_7;
        tt::tt_metal::CircularBufferConfig in_mask_cb_config =
            tt::tt_metal::CircularBufferConfig(in_mask_CB_size, {{in_mask_cb_index, in_mask_cb_data_format}})
                .set_page_size(in_mask_cb_index, in_mask_single_tile_size);
        auto cb_inz = tt::tt_metal::CreateCircularBuffer(program, all_cores, in_mask_cb_config);
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
        auto cb_inz = tt::tt_metal::CreateCircularBuffer(program, all_cores, repack_cb_config);
    }
    // x
    uint32_t x_cb_index = tt::CBIndex::c_13;
    tt::tt_metal::CircularBufferConfig x_cb_config =
        tt::tt_metal::CircularBufferConfig(x_CB_size, {{x_cb_index, cb_data_format}})
            .set_page_size(x_cb_index, single_tile_size);
    auto cb_x = tt::tt_metal::CreateCircularBuffer(program, all_cores, x_cb_config);
    // xmm
    uint32_t xmm_cb_index = tt::CBIndex::c_14;
    tt::tt_metal::CircularBufferConfig xmm_cb_config =
        tt::tt_metal::CircularBufferConfig(xmm_CB_size, {{xmm_cb_index, cb_data_format}})
            .set_page_size(xmm_cb_index, single_tile_size);
    auto cb_xmm = tt::tt_metal::CreateCircularBuffer(program, all_cores, xmm_cb_config);
    // ex_partial
    uint32_t ex_cb_partial_index = tt::CBIndex::c_8;
    tt::tt_metal::CircularBufferConfig ex_cb_partial_config =
        tt::tt_metal::CircularBufferConfig(ex_partial_CB_size, {{ex_cb_partial_index, cb_data_format}})
            .set_page_size(ex_cb_partial_index, single_tile_size);
    auto cb_ex_partial = tt::tt_metal::CreateCircularBuffer(program, all_cores, ex_cb_partial_config);
    // ex_external
    uint32_t ex_cb_external_index = tt::CBIndex::c_10;
    tt::tt_metal::CircularBufferConfig ex_cb_external_config =
        tt::tt_metal::CircularBufferConfig(
            single_tile_size * num_cores_per_mcast_group, {{ex_cb_external_index, cb_data_format}})
            .set_page_size(ex_cb_external_index, single_tile_size);
    auto cb_ex_external = tt::tt_metal::CreateCircularBuffer(program, all_cores, ex_cb_external_config);
    // ex_global
    uint32_t ex_cb_index = tt::CBIndex::c_9;
    uint32_t ex_global_cb_index = tt::CBIndex::c_15;
    std::map<uint8_t, tt::DataFormat> ex_global_cb_data_format_spec{
        {ex_global_cb_index, cb_data_format}, {ex_cb_index, cb_data_format}};
    auto ex_global_cb_config = tt::tt_metal::CircularBufferConfig(ex_global_CB_size, ex_global_cb_data_format_spec)
                                   .set_page_size(ex_global_cb_index, single_tile_size)
                                   .set_page_size(ex_cb_index, single_tile_size);
    auto cb_ex_global = tt::tt_metal::CreateCircularBuffer(program, all_cores, ex_global_cb_config);
    // ex2pe
    uint32_t cb_ex2pe_index;
    cb_ex2pe_index = tt::CBIndex::c_17;
    tt::tt_metal::CircularBufferConfig ex2pe_cb_config =
        tt::tt_metal::CircularBufferConfig(ex2pe_CB_size, {{cb_ex2pe_index, cb_data_format}})
            .set_page_size(cb_ex2pe_index, single_tile_size);
    auto cb_ex2pe = tt::tt_metal::CreateCircularBuffer(program, all_cores, ex2pe_cb_config);

    // Runtime Args
    std::vector<KernelHandle> writer_kernel_ids;
    float winv = 1.0f / std::sqrt(num_rows_per_batch_per_core * num_datum_row_per_group);  // bcast-w scaler
    bfloat16 bfloat_winv_value = bfloat16(winv);
    uint32_t packed_winv_value = pack_two_bfloat16_into_uint32({bfloat_winv_value, bfloat_winv_value});
    float cinv = 1.0f / std::sqrt(num_cores_per_batch * num_cores_per_group);  // bcast-cores scaler
    bfloat16 bfloat_cinv_value = bfloat16(cinv);
    uint32_t packed_cinv_value = pack_two_bfloat16_into_uint32({bfloat_cinv_value, bfloat_cinv_value});
    union {
        float f;
        uint32_t u;
    } e;
    e.f = eps;

    log_debug(tt::LogOp, "num_rows_per_batch_per_core: {}", num_rows_per_batch_per_core);
    log_debug(tt::LogOp, "num_datum_row_per_group: {}", num_datum_row_per_group);
    log_debug(tt::LogOp, "num_cores_per_batch: {}", num_cores_per_batch);
    log_debug(tt::LogOp, "num_cores_per_group: {}", num_cores_per_group);

    for (int i = 0; i < mcast_groups.size(); ++i) {
        auto group = mcast_groups[i];
        bool rectangle_grid = is_rectangle_grid(group);

        for (int j = 0; j < group.size(); ++j) {
            CoreCoord core = group[j];
            CoreCoord core_physical = device->worker_core_from_logical_core(core);

            if (j == 0) {  // mcast sender
                // get the bounding box for the mcast
                std::vector<CoreCoord> mcast_group_first;
                std::vector<CoreCoord> mcast_group_mid(group);
                std::vector<CoreCoord> mcast_group_last;
                if (not rectangle_grid) {
                    split_and_form_rectangle_grids(group, mcast_group_first, mcast_group_mid, mcast_group_last);
                }

                CoreCoord mcast_start = device->worker_core_from_logical_core(mcast_group_mid.front());
                CoreCoord mcast_end = device->worker_core_from_logical_core(mcast_group_mid.back());

                if (reader_noc == NOC::NOC_1) {
                    std::swap(mcast_start, mcast_end);
                }
                std::vector<uint32_t> mcast_sender_args;
                mcast_sender_args.push_back(not mcast_group_first.empty());
                mcast_sender_args.push_back(not mcast_group_last.empty());
                mcast_sender_args.push_back(mcast_start.x);
                mcast_sender_args.push_back(mcast_start.y);
                mcast_sender_args.push_back(mcast_end.x);
                mcast_sender_args.push_back(mcast_end.y);
                if (not mcast_group_first.empty()) {
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

                if (not mcast_group_first.empty()) {
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
                if (not mcast_group_last.empty()) {
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
                for (int c = 0; c < group.size(); ++c) {
                    CoreCoord coord = device->worker_core_from_logical_core(group[c]);
                    mcast_noc_xy.push_back(coord.x);
                }
                for (int c = 0; c < group.size(); ++c) {
                    CoreCoord coord = device->worker_core_from_logical_core(group[c]);
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
    for (int i = 0; i < core_coords.size(); ++i) {
        auto core = core_coords[i];

        std::vector<uint32_t> writer_mcast_sender_args;
        writer_mcast_sender_args.push_back(packed_cinv_value);
        writer_mcast_sender_args.push_back(packed_winv_value);
        writer_mcast_sender_args.push_back(e.u);
        writer_mcast_sender_args.push_back(gamma_dram_addr);
        writer_mcast_sender_args.push_back(beta_dram_addr);
        writer_mcast_sender_args.push_back(input_mask_dram_addr);
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
            input_mask_tile_start_id = (input_mask_tile_start_id + input_mask_num_tiles_per_core) %
                                       (input_mask.value().physical_volume() / TILE_HW);
        }
    }

    auto override_runtime_args_callback = [writer_kernel_ids, cb_in0, cb_output, num_cores, grid_size](
                                              const void* operation,
                                              Program& program,
                                              const std::vector<Tensor>& input_tensors,
                                              const std::vector<std::optional<const Tensor>>& optional_input_tensors,
                                              const std::vector<Tensor>& output_tensors) {
        auto src_buffer_a = input_tensors.at(0).buffer();
        auto gamma_tensor = optional_input_tensors.at(0);
        auto beta_tensor = optional_input_tensors.at(1);
        auto mask_tensor = optional_input_tensors.at(2);
        auto dst_buffer = output_tensors.at(0).buffer();

        UpdateDynamicCircularBufferAddress(program, cb_in0, *src_buffer_a);
        UpdateDynamicCircularBufferAddress(program, cb_output, *dst_buffer);

        for (uint32_t i = 0; i < num_cores; ++i) {
            CoreCoord core = {i % grid_size.x, i / grid_size.x};

            auto writer_kernel_id = writer_kernel_ids.at(i);

            auto& runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);

            if (gamma_tensor.has_value()) {
                runtime_args[3] = gamma_tensor.value().buffer()->address();
            }
            if (beta_tensor.has_value()) {
                runtime_args[4] = beta_tensor.value().buffer()->address();
            }
            if (mask_tensor.has_value()) {
                runtime_args[5] = mask_tensor.value().buffer()->address();
            }
        }
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_args_callback};
}

operation::ProgramWithCallbacks groupnorm_multi_core(
    const Tensor& a,
    const std::optional<const Tensor>& gamma,
    const std::optional<const Tensor>& beta,
    const std::optional<const Tensor>& input_mask,
    Tensor& output,
    float eps,
    const uint32_t num_groups,
    const uint32_t num_batches,
    MathFidelity fidelity,
    DataType im_data_format,
    CoreCoord grid_size,
    bool inplace,
    uint32_t num_out_blocks) {
    using namespace CMAKE_UNIQUE_NAMESPACE;

    if (gamma.has_value()) {
        TT_FATAL(gamma.value().layout() == Layout::ROW_MAJOR, "Gamma tensor must have ROW_MAJOR layout");
    }
    if (beta.has_value()) {
        TT_FATAL(beta.value().layout() == Layout::ROW_MAJOR, "Beta tensor must have ROW_MAJOR layout");
    }

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
    uint32_t datum_size_bytes = 2;  // bfloat16

    TT_FATAL(
        out_data_format == in_data_format,
        "input: {} and output: {} must be the same data format",
        in_data_format,
        out_data_format);

    // tile sizes
    uint32_t in_single_tile_size = tt::tt_metal::detail::TileSize(in_data_format);
    uint32_t single_tile_size = tt::tt_metal::detail::TileSize(cb_data_format);
    uint32_t out_single_tile_size = tt::tt_metal::detail::TileSize(out_data_format);
    uint32_t gamma_beta_single_tile_size = tt::tt_metal::detail::TileSize(gamma_beta_cb_data_format);
    uint32_t in_mask_single_tile_size = tt::tt_metal::detail::TileSize(in_mask_cb_data_format);

    IDevice* device = a.device();

    // grid
    uint32_t num_cores_c = grid_size.y;
    uint32_t num_cores_r = grid_size.x;
    uint32_t num_cores = num_cores_c * num_cores_r;
    auto all_cores = tt::tt_metal::num_cores_to_corerangeset(num_cores, grid_size, true);

    // tensor shape
    const auto shape = a.padded_shape();
    uint32_t H = shape[2] * num_batches;
    uint32_t Ht = H / TILE_HEIGHT;
    uint32_t W = shape[3];
    uint32_t Wt = W / TILE_WIDTH;
    uint32_t per_core_M_group_1 = H / num_cores_r;
    uint32_t per_core_M_group_2 = 0;
    uint32_t per_core_N = W / num_cores_c;
    TT_FATAL(num_cores_c != 0, "num_cores_c should not equal 0");
    TT_FATAL(num_cores_r != 0, "num_cores_r should not equal 0");
    TT_FATAL(H % num_cores_r == 0, "width * height: {} must be divisible by num_cores.y: {}", H, num_cores_r);
    TT_FATAL(W % num_cores_c == 0, "channels: {} must be divisible by num_cores.x: {}", W, num_cores_c);
    uint32_t per_core_Mt_group_1 = per_core_M_group_1 / TILE_HEIGHT;
    uint32_t per_core_Mt_group_2 = 0;
    uint32_t per_core_Nt = (per_core_N + TILE_WIDTH - 1) / TILE_WIDTH;
    uint32_t num_datum_row_per_group = W / num_groups;
    uint32_t num_datum_row_per_group_mod_tile_w =
        num_datum_row_per_group % TILE_WIDTH == 0 ? TILE_WIDTH : num_datum_row_per_group % TILE_WIDTH;
    uint32_t group_size = W / num_groups;
    // split each batch into multiple cores
    uint32_t num_shards_r = H / per_core_M_group_1;
    uint32_t num_cores_per_batch = num_batches > num_shards_r ? 1 : num_shards_r / num_batches;
    uint32_t num_shards_c = W / per_core_N;
    uint32_t num_cores_per_group = num_groups > num_shards_c ? 1 : num_shards_c / num_groups;
    // each core contains multiple batches
    uint32_t num_batches_per_core_group_1 = num_batches > num_shards_r ? num_batches / num_shards_r : 1;
    uint32_t num_batches_per_core_group_2 = num_batches_per_core_group_1;  // need this to be non-zero even if unused
    uint32_t num_groups_per_core = num_groups > num_shards_c ? num_groups / num_shards_c : 1;
    TT_FATAL(num_groups % num_cores_r == 0, "num_groups: {} must divide cores_y: {}", num_groups, num_cores_r);
    TT_FATAL(
        (num_groups / num_cores_r) * group_size % TILE_WIDTH == 0,
        "(num_groups: {}/cores_x: {})*(num_channels: {}/num_groups: {}) must be divisible by {}",
        num_groups,
        num_cores_r,
        W,
        num_groups,
        TILE_WIDTH);

    // subblock
    uint32_t num_rows_per_batch_per_core_group_1 = per_core_M_group_1 / num_batches_per_core_group_1;
    uint32_t num_rows_per_batch_per_core_group_2 = 0;
    auto [block_wt, num_groups_per_reset] = find_max_tile_span(per_core_N, group_size);
    uint32_t block_ht_group_1 = per_core_Mt_group_1 / num_batches_per_core_group_1;
    uint32_t block_ht_group_2 = 0;
    uint32_t subblock_wt = get_max_subblock(block_wt, 8);
    uint32_t num_subblocks_w = block_wt / subblock_wt;
    bool block_wt_last = (per_core_Nt + num_groups_per_core - 1) / num_groups_per_core;

    // support for uneven batches across rows
    bool equal_batches_per_core = true;
    uint32_t last_row_with_extra_batch = 0;
    if (num_batches >= num_cores_r) {
        last_row_with_extra_batch = (num_batches % num_shards_r);
        equal_batches_per_core = (last_row_with_extra_batch == 0);
        if (!equal_batches_per_core) {
            last_row_with_extra_batch--;  // zero based index
        }
    }

    // Have first group (each row has 1 extra batch compared to second group), and second group
    if (!equal_batches_per_core) {
        // tensor shape
        num_batches_per_core_group_2 = num_batches / num_cores_r;
        num_batches_per_core_group_1 = num_batches_per_core_group_2 + 1;

        TT_FATAL(Ht % num_batches == 0, "Ht ({}) needs to be divisible by the number of batches ({})", Ht, num_batches);
        uint32_t per_batch_tiles = Ht / num_batches;
        per_core_Mt_group_1 = num_batches_per_core_group_1 * per_batch_tiles;
        per_core_Mt_group_2 = num_batches_per_core_group_2 * per_batch_tiles;
        per_core_M_group_1 = per_core_Mt_group_1 * TILE_HEIGHT;
        per_core_M_group_2 = per_core_Mt_group_2 * TILE_HEIGHT;

        // subblock
        num_rows_per_batch_per_core_group_1 = per_core_M_group_1 / num_batches_per_core_group_1;
        num_rows_per_batch_per_core_group_2 = per_core_M_group_2 / num_batches_per_core_group_2;
        block_ht_group_1 = per_core_Mt_group_1 / num_batches_per_core_group_1;
        block_ht_group_2 = per_core_Mt_group_2 / num_batches_per_core_group_2;
    }

    // shard shape per core
    uint32_t per_core_N_bytes_padded = tt::round_up(per_core_N * datum_size_bytes, output.buffer()->alignment());
    bool reader_repack_output = (per_core_N % TILE_WIDTH) != 0;
    bool tilize_in = a.layout() == Layout::ROW_MAJOR;
    bool untilize_out = output.layout() == Layout::ROW_MAJOR;

    TT_FATAL(
        per_core_N % num_datum_row_per_group == 0,
        "per_core_N ({}) must be divisible by num_datum_row_per_group ({})",
        per_core_N,
        num_datum_row_per_group);
    TT_FATAL(num_datum_row_per_group != 0, "num_datum_row_per_group should not equal 0");
    TT_FATAL(per_core_M_group_1 % TILE_HEIGHT == 0, "per_core_M: {} divides Tile Height", per_core_M_group_1);
    if (per_core_M_group_2 > 0) {
        TT_FATAL(per_core_M_group_2 % TILE_HEIGHT == 0, "per_core_M: {} divides Tile Height", per_core_M_group_2);
    }
    if (per_core_N != W) {
        TT_FATAL(per_core_N * num_cores_c == W, "cores_x mus divide Channels");
        // TT_FATAL(per_core_M_group_1 * num_cores_r == H, "{} * {} should equal {}", per_core_M_group_1, num_cores_r,
        // H); TODO VASH
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
    log_debug(tt::LogOp, "num_datum_row_per_group: {}", num_datum_row_per_group);
    log_debug(tt::LogOp, "num_batches: {}", num_batches);
    log_debug(tt::LogOp, "num_groups: {}", num_groups);
    log_debug(tt::LogOp, "num_cores_r: {}", num_cores_r);
    log_debug(tt::LogOp, "num_cores_c: {}", num_cores_c);
    log_debug(tt::LogOp, "num_cores_per_batch: {}", num_cores_per_batch);
    log_debug(tt::LogOp, "num_cores_per_group: {}", num_cores_per_group);
    log_debug(tt::LogOp, "num_batches_per_core_group_1: {}", num_batches_per_core_group_1);
    log_debug(tt::LogOp, "num_batches_per_core_group_2: {}", num_batches_per_core_group_2);
    log_debug(tt::LogOp, "equal_batches_per_core: {}", equal_batches_per_core);
    log_debug(tt::LogOp, "num_groups_per_core: {}", num_groups_per_core);
    log_debug(tt::LogOp, "block_wt: {}", block_wt);
    log_debug(tt::LogOp, "block_wt_last: {}", block_wt_last);
    log_debug(tt::LogOp, "block_ht_group_1: {}", block_ht_group_1);
    log_debug(tt::LogOp, "block_ht_group_2: {}", block_ht_group_2);
    log_debug(tt::LogOp, "subblock_wt: {}", subblock_wt);
    log_debug(tt::LogOp, "num_subblocks_w: {}", num_subblocks_w);
    log_debug(tt::LogOp, "reader_repack_output: {}", reader_repack_output);

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
    // gamma, beta addr
    auto gamma_dram_addr = gamma.has_value() ? gamma.value().buffer()->address() : 0;
    auto beta_dram_addr = beta.has_value() ? beta.value().buffer()->address() : 0;
    auto input_mask_dram_addr = input_mask.has_value() ? input_mask.value().buffer()->address() : 0;
    // num tiles for a, gamma, beta
    uint32_t num_tiles = a.physical_volume() / TILE_HW;
    uint32_t num_gamma_tiles = gamma.has_value() ? gamma.value().physical_volume() / TILE_HW : 0;
    uint32_t num_beta_tiles = beta.has_value() ? beta.value().physical_volume() / TILE_HW : 0;
    uint32_t num_input_mask_tiles = input_mask.has_value() ? input_mask.value().physical_volume() / TILE_HW : 0;

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    // block size for in0 (tensor a)
    uint32_t in0_block_tiles_group_1 = block_ht_group_1 / num_out_blocks * block_wt;
    uint32_t in0_block_tiles_group_2 = 0;
    uint32_t in0_CB_size_group_1 = in0_block_tiles_group_1 * in_single_tile_size;
    uint32_t in0_CB_size_group_2 = 0;
    uint32_t in_CB_size_group_1 = in0_block_tiles_group_1 * in_single_tile_size;
    uint32_t in_CB_size_group_2 = 0;
    // in2 - scaler
    uint32_t in2_CB_size = single_tile_size;
    // in3 - eps
    uint32_t in3_CB_size = single_tile_size;
    // gamma
    uint32_t gamma_beta_num_cols_tile_per_core = per_core_Nt;
    uint32_t in5_CB_size = gamma_beta_num_cols_tile_per_core * gamma_beta_single_tile_size;
    // beta
    uint32_t in6_CB_size = gamma_beta_num_cols_tile_per_core * gamma_beta_single_tile_size;
    // input mask
    uint32_t input_mask_num_tiles_per_core = block_wt * num_groups_per_core;
    uint32_t in_mask_CB_size = block_wt * in_mask_single_tile_size * 2;  // double buffer
    // repack cb
    uint32_t repack_CB_size = per_core_Nt * in_single_tile_size * 2;  // double buffer
    // itermediate buffers
    uint32_t interm_block_tiles_group_1 = block_ht_group_1 / num_out_blocks * block_wt;
    uint32_t interm_block_tiles_group_2 = 0;
    uint32_t x_CB_size_group_1 = interm_block_tiles_group_1 * single_tile_size;
    uint32_t x_CB_size_group_2 = 0;
    uint32_t xmm_CB_size_group_1 = interm_block_tiles_group_1 * single_tile_size;
    uint32_t xmm_CB_size_group_2 = 0;
    uint32_t ex_partial_CB_size = single_tile_size;   // partial Ex
    uint32_t ex2_partial_CB_size = single_tile_size;  // partial Ex2
    uint32_t ex_global_CB_size = ex_partial_CB_size;  // the final result Ex
    uint32_t ex2_global_CB_size = ex2_partial_CB_size;  // the final result Ex2
    uint32_t xmm2_CB_size_group_1 = interm_block_tiles_group_1 * single_tile_size;
    uint32_t xmm2_CB_size_group_2 = 0;
    uint32_t xmm3_CB_size_group_1 = interm_block_tiles_group_1 * single_tile_size;
    uint32_t xmm3_CB_size_group_2 = 0;
    uint32_t ex2pe_CB_size = ex_partial_CB_size;
    // output buffer size
    uint32_t out_CB_size_group_1 = in0_block_tiles_group_1 * out_single_tile_size;
    uint32_t out_CB_size_group_2 = 0;

    if (!equal_batches_per_core) {
        // input buffers
        in0_block_tiles_group_1 = block_ht_group_1 / num_out_blocks * block_wt;
        in0_block_tiles_group_2 = block_ht_group_2 / num_out_blocks * block_wt;
        in0_CB_size_group_1 = in0_block_tiles_group_1 * in_single_tile_size;
        in0_CB_size_group_2 = in0_block_tiles_group_2 * in_single_tile_size;
        in_CB_size_group_1 = in0_block_tiles_group_1 * in_single_tile_size;
        in_CB_size_group_2 = in0_block_tiles_group_2 * in_single_tile_size;
        // intermediate buffers
        interm_block_tiles_group_1 = block_ht_group_1 / num_out_blocks * block_wt;
        interm_block_tiles_group_2 = block_ht_group_2 / num_out_blocks * block_wt;
        x_CB_size_group_1 = interm_block_tiles_group_1 * single_tile_size;
        x_CB_size_group_2 = interm_block_tiles_group_2 * single_tile_size;
        xmm_CB_size_group_1 = interm_block_tiles_group_1 * single_tile_size;
        xmm_CB_size_group_2 = interm_block_tiles_group_2 * single_tile_size;
        xmm2_CB_size_group_1 = interm_block_tiles_group_1 * single_tile_size;
        xmm2_CB_size_group_2 = interm_block_tiles_group_2 * single_tile_size;
        xmm3_CB_size_group_1 = interm_block_tiles_group_1 * single_tile_size;
        xmm3_CB_size_group_2 = interm_block_tiles_group_2 * single_tile_size;
        // output buffer size
        out_CB_size_group_1 = in0_block_tiles_group_1 * out_single_tile_size;
        out_CB_size_group_2 = in0_block_tiles_group_2 * out_single_tile_size;
    }

    // Do CB size check with group_2 since it's larger
    // if (equal_batches_per_core) {
    //     TT_FATAL(
    //         cbs_fit_in_DRAM(
    //             in0_CB_size_group_1,
    //             in_CB_size_group_1,
    //             in2_CB_size,
    //             in3_CB_size,
    //             in5_CB_size,
    //             in6_CB_size,
    //             in_mask_CB_size,
    //             repack_CB_size,
    //             x_CB_size_group_1,
    //             xmm_CB_size_group_1,
    //             ex_partial_CB_size,
    //             ex_global_CB_size,
    //             ex2_global_CB_size,
    //             xmm2_CB_size_group_1,
    //             xmm3_CB_size_group_1,
    //             ex2pe_CB_size,
    //             out_CB_size_group_1,
    //             a.device()->l1_size_per_core()),
    //         "Circular buffers require too much space to fit into L1");
    // } else {
    //     TT_FATAL(
    //         cbs_fit_in_DRAM(
    //             in0_CB_size_group_2,
    //             in_CB_size_group_2,
    //             in2_CB_size,
    //             in3_CB_size,
    //             in5_CB_size,
    //             in6_CB_size,
    //             in_mask_CB_size,
    //             repack_CB_size,
    //             x_CB_size_group_2,
    //             xmm_CB_size_group_2,
    //             ex_partial_CB_size,
    //             ex_global_CB_size,
    //             ex2_global_CB_size,
    //             xmm2_CB_size_group_2,
    //             xmm3_CB_size_group_2,
    //             ex2pe_CB_size,
    //             out_CB_size_group_2,
    //             a.device()->l1_size_per_core()),
    //         "Circular buffers require too much space to fit into L1");
    // }

    log_debug(tt::LogOp, "per_core_Nt: {}", per_core_Nt);
    log_debug(tt::LogOp, "per_core_Mt_group_1: {}", per_core_Mt_group_1);
    log_debug(tt::LogOp, "per_core_Mt_group_2: {}", per_core_Mt_group_2);
    log_debug(tt::LogOp, "in0_CB_size_group_1: {}", in0_CB_size_group_1);
    log_debug(tt::LogOp, "in0_CB_size_group_2: {}", in0_CB_size_group_2);
    log_debug(tt::LogOp, "in_CB_size_group_1: {}", in_CB_size_group_1);
    log_debug(tt::LogOp, "in_CB_size_group_2: {}", in_CB_size_group_2);
    log_debug(tt::LogOp, "gamma_beta_num_cols_tile_per_core: {}", gamma_beta_num_cols_tile_per_core);
    log_debug(tt::LogOp, "in5_CB_size: {}", in5_CB_size);
    log_debug(tt::LogOp, "repack_CB_size: {}", repack_CB_size);

    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    Program program = Program();
    // define core ranges
    bool use_mcast = num_cores_per_batch > 1 or num_cores_per_group > 1;
    uint32_t start_core_x = 0;
    uint32_t start_core_y = 0;

    // create a vector of cores, in either RM or CM
    std::vector<CoreCoord> core_coords = grid_to_cores(num_cores, num_cores_r, num_cores_c, false);
    for (int i = 0; i < core_coords.size(); ++i) {
        log_debug(tt::LogOp, "worker coord: {} {}", core_coords[i].x, core_coords[i].y);
    }
    std::set<CoreRange> all_cores_group_1_core_ranges;
    std::set<CoreRange> all_cores_group_2_core_ranges;
    for (int i = 0; i < num_cores; ++i) {
        CoreCoord core = core_coords[i];
        if (equal_batches_per_core || (core.x <= last_row_with_extra_batch)) {
            all_cores_group_1_core_ranges.insert(CoreRange(core_coords[i]));
        } else {
            all_cores_group_2_core_ranges.insert(CoreRange(core_coords[i]));
        }
    }
    CoreRangeSet all_cores_group_1 = CoreRangeSet(all_cores_group_1_core_ranges);
    CoreRangeSet all_cores_group_2 = CoreRangeSet(all_cores_group_2_core_ranges);

    std::vector<std::vector<CoreCoord>> core_coords2D;
    for (int j = 0; j < num_cores_c; ++j) {
        for (int i = 0; i < num_cores_r / num_cores_per_group; ++i) {
            std::vector<CoreCoord> temp;
            for (int k = 0; k < num_cores_per_group; ++k) {
                temp.push_back(CoreCoord{(std::size_t)(k + i * num_cores_per_group), (std::size_t)j});
            }
            core_coords2D.push_back(temp);
        }
    }

    // one mcast core per batch per group
    std::set<CoreRange> mcast_sender_core_ranges_group_1;
    std::set<CoreRange> mcast_sender_core_ranges_group_2;
    std::set<CoreRange> mcast_sender_core_ranges_all;
    std::set<CoreRange> mcast_receiver_core_ranges_group_1;
    std::set<CoreRange> mcast_receiver_core_ranges_group_2;
    std::set<CoreRange> mcast_receiver_core_ranges_all;
    uint32_t core_index = 0;
    uint32_t core_index_offset = 0;
    uint32_t sender_groups_count = equal_batches_per_core ? (num_batches / num_batches_per_core_group_1) : num_cores_r;
    for (int i = 0; i < sender_groups_count; ++i) {
        uint32_t core_index = core_index_offset;
        for (int j = 0; j < num_groups / num_groups_per_core; ++j) {
            mcast_sender_core_ranges_all.insert(CoreRange(core_coords[core_index]));
            CoreCoord core = core_coords[core_index];
            if (equal_batches_per_core || (core.x <= last_row_with_extra_batch)) {
                mcast_sender_core_ranges_group_1.insert(CoreRange(core_coords[core_index]));
            } else {
                mcast_sender_core_ranges_group_2.insert(CoreRange(core_coords[core_index]));
            }
            core_index += num_cores_per_group;
            core_index_offset += num_cores_per_batch * num_cores_per_group;
        }
    }
    for (auto& coord : mcast_sender_core_ranges_all) {
        log_debug(tt::LogOp, "mcast sender coord: {} {}", coord.start_coord.x, coord.start_coord.y);
    }
    for (auto& coord : mcast_sender_core_ranges_group_1) {
        log_debug(tt::LogOp, "mcast sender coord group 1: {} {}", coord.start_coord.x, coord.start_coord.y);
    }
    for (auto& coord : mcast_sender_core_ranges_group_2) {
        log_debug(tt::LogOp, "mcast sender coord group 2: {} {}", coord.start_coord.x, coord.start_coord.y);
    }
    for (int i = 0; i < num_cores; ++i) {
        // not found in mcast sender
        if (mcast_sender_core_ranges_all.find(CoreRange(core_coords[i])) == mcast_sender_core_ranges_all.end()) {
            mcast_receiver_core_ranges_all.insert(CoreRange(core_coords[i]));
            CoreCoord core = core_coords[i];
            if (equal_batches_per_core || (core.x <= last_row_with_extra_batch)) {
                mcast_receiver_core_ranges_group_1.insert(CoreRange(core_coords[i]));
            } else {
                mcast_receiver_core_ranges_group_2.insert(CoreRange(core_coords[i]));
            }
        }
    }
    for (auto& coord : mcast_receiver_core_ranges_all) {
        log_debug(tt::LogOp, "mcast receiver coord: {} {}", coord.start_coord.x, coord.start_coord.y);
    }
    for (auto& coord : mcast_receiver_core_ranges_group_1) {
        log_debug(tt::LogOp, "mcast receiver coord group 1: {} {}", coord.start_coord.x, coord.start_coord.y);
    }
    for (auto& coord : mcast_receiver_core_ranges_group_2) {
        log_debug(tt::LogOp, "mcast receiver coord group 2: {} {}", coord.start_coord.x, coord.start_coord.y);
    }
    CoreRangeSet mcast_sender_cores_group_1 = CoreRangeSet(mcast_sender_core_ranges_group_1);
    CoreRangeSet mcast_receiver_cores_group_1 = CoreRangeSet(mcast_receiver_core_ranges_group_1);
    CoreRangeSet mcast_sender_cores_group_2 = CoreRangeSet(mcast_sender_core_ranges_group_2);
    CoreRangeSet mcast_receiver_cores_group_2 = CoreRangeSet(mcast_receiver_core_ranges_group_2);
    // mcast groups
    std::vector<std::vector<CoreCoord>> mcast_groups;
    int group_index = -1;
    for (int i = 0; i < core_coords2D.size(); ++i) {
        for (int j = 0; j < core_coords2D[i].size(); ++j) {
            if (mcast_sender_core_ranges_all.find(CoreRange(core_coords2D[i][j])) !=
                mcast_sender_core_ranges_all.end()) {
                group_index += 1;
            }
            if (group_index >= mcast_groups.size()) {
                mcast_groups.push_back(std::vector<CoreCoord>());  // Add a new group
            }
            mcast_groups[group_index].push_back(core_coords2D[i][j]);
        }
    }
    for (int i = 0; i < mcast_groups.size(); ++i) {
        for (int j = 0; j < mcast_groups[i].size(); ++j) {
            log_debug(tt::LogOp, "mcast group: {} coord: {} {}", i, mcast_groups[i][j].x, mcast_groups[i][j].y);
        }
    }
    // how many cores in a mcast group
    uint32_t num_cores_per_mcast_group = mcast_groups[0].size();
    // Mcast args
    auto reduce_sender_semaphore_id = tt::tt_metal::CreateSemaphore(program, all_cores, INVALID);
    auto reduce_receiver_semaphore_id = tt::tt_metal::CreateSemaphore(program, all_cores, INVALID);
    // reader defines
    std::map<string, string> reader_mcast_sender_defines;
    std::map<string, string> reader_mcast_receiver_defines;
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
    std::vector<uint32_t> reader_mcast_sender_compile_time_args_group_1 = {
        (std::uint32_t)1,
        (std::uint32_t)1,
        (std::uint32_t)reduce_receiver_semaphore_id,
        (std::uint32_t)reduce_sender_semaphore_id,
        (std::uint32_t)num_cores_per_mcast_group,
        (std::uint32_t)num_groups_per_core * num_batches_per_core_group_1,
        (std::uint32_t)num_batches_per_core_group_1,
        (std::uint32_t)per_core_Nt,
        (std::uint32_t)per_core_N_bytes_padded,
        (std::uint32_t)per_core_Nt * TILE_WIDTH * datum_size_bytes,
        (std::uint32_t)datum_size_bytes,
        (std::uint32_t)per_core_Mt_group_1,
        (std::uint32_t)TILE_HEIGHT,
        (std::uint32_t)block_ht_group_1,
        (std::uint32_t)block_wt,
        (std::uint32_t)block_ht_group_1 * block_wt,
        (std::uint32_t)num_datum_row_per_group_mod_tile_w,
        (std::uint32_t)per_core_Mt_group_1 * Wt / num_batches_per_core_group_1,
        (std::uint32_t)block_wt_last,
        (std::uint32_t)(num_datum_row_per_group_mod_tile_w & (num_datum_row_per_group_mod_tile_w - 1)) == 0,
        (std::uint32_t)num_datum_row_per_group < TILE_WIDTH,
        (std::uint32_t)num_datum_row_per_group - (block_wt - 1) * TILE_WIDTH,
        (std::uint32_t)num_out_blocks};
    std::vector<uint32_t> reader_mcast_receiver_compile_time_args_group_1 = {
        (std::uint32_t)1,
        (std::uint32_t)1,
        (std::uint32_t)reduce_receiver_semaphore_id,
        (std::uint32_t)reduce_sender_semaphore_id,
        (std::uint32_t)num_groups_per_core * num_batches_per_core_group_1,
        (std::uint32_t)num_batches_per_core_group_1,
        (std::uint32_t)per_core_Nt,
        (std::uint32_t)per_core_N_bytes_padded,
        (std::uint32_t)per_core_Nt * TILE_WIDTH * datum_size_bytes,
        (std::uint32_t)per_core_Mt_group_1,
        (std::uint32_t)TILE_HEIGHT,
        (std::uint32_t)block_ht_group_1,
        (std::uint32_t)block_wt,
        (std::uint32_t)block_ht_group_1 * block_wt,
        (std::uint32_t)num_datum_row_per_group_mod_tile_w,
        (std::uint32_t)per_core_Mt_group_1 * Wt / num_batches_per_core_group_1,
        (std::uint32_t)block_wt_last,
        (std::uint32_t)(num_datum_row_per_group_mod_tile_w & (num_datum_row_per_group_mod_tile_w - 1)) == 0,
        (std::uint32_t)num_datum_row_per_group < TILE_WIDTH,
        (std::uint32_t)num_datum_row_per_group - (block_wt - 1) * TILE_WIDTH,
        (std::uint32_t)num_out_blocks};
    std::vector<uint32_t> reader_mcast_sender_compile_time_args_group_2 = {
        (std::uint32_t)1,
        (std::uint32_t)1,
        (std::uint32_t)reduce_receiver_semaphore_id,
        (std::uint32_t)reduce_sender_semaphore_id,
        (std::uint32_t)num_cores_per_mcast_group,
        (std::uint32_t)num_groups_per_core * num_batches_per_core_group_2,
        (std::uint32_t)num_batches_per_core_group_2,
        (std::uint32_t)per_core_Nt,
        (std::uint32_t)per_core_N_bytes_padded,
        (std::uint32_t)per_core_Nt * TILE_WIDTH * datum_size_bytes,
        (std::uint32_t)datum_size_bytes,
        (std::uint32_t)per_core_Mt_group_2,
        (std::uint32_t)TILE_HEIGHT,
        (std::uint32_t)block_ht_group_2,
        (std::uint32_t)block_wt,
        (std::uint32_t)block_ht_group_2 * block_wt,
        (std::uint32_t)num_datum_row_per_group_mod_tile_w,
        (std::uint32_t)per_core_Mt_group_2 * Wt / num_batches_per_core_group_2,
        (std::uint32_t)block_wt_last,
        (std::uint32_t)(num_datum_row_per_group_mod_tile_w & (num_datum_row_per_group_mod_tile_w - 1)) == 0,
        (std::uint32_t)num_datum_row_per_group < TILE_WIDTH,
        (std::uint32_t)num_datum_row_per_group - (block_wt - 1) * TILE_WIDTH,
        (std::uint32_t)num_out_blocks};
    std::vector<uint32_t> reader_mcast_receiver_compile_time_args_group_2 = {
        (std::uint32_t)1,
        (std::uint32_t)1,
        (std::uint32_t)reduce_receiver_semaphore_id,
        (std::uint32_t)reduce_sender_semaphore_id,
        (std::uint32_t)num_groups_per_core * num_batches_per_core_group_2,
        (std::uint32_t)num_batches_per_core_group_2,
        (std::uint32_t)per_core_Nt,
        (std::uint32_t)per_core_N_bytes_padded,
        (std::uint32_t)per_core_Nt * TILE_WIDTH * datum_size_bytes,
        (std::uint32_t)per_core_Mt_group_2,
        (std::uint32_t)TILE_HEIGHT,
        (std::uint32_t)block_ht_group_2,
        (std::uint32_t)block_wt,
        (std::uint32_t)block_ht_group_2 * block_wt,
        (std::uint32_t)num_datum_row_per_group_mod_tile_w,
        (std::uint32_t)per_core_Mt_group_2 * Wt / num_batches_per_core_group_2,
        (std::uint32_t)block_wt_last,
        (std::uint32_t)(num_datum_row_per_group_mod_tile_w & (num_datum_row_per_group_mod_tile_w - 1)) == 0,
        (std::uint32_t)num_datum_row_per_group < TILE_WIDTH,
        (std::uint32_t)num_datum_row_per_group - (block_wt - 1) * TILE_WIDTH,
        (std::uint32_t)num_out_blocks};
    tt::tt_metal::NOC reader_noc = tt::tt_metal::detail::GetPreferredNOCForDRAMWrite(device->arch());
    tt::tt_metal::NOC writer_noc = tt::tt_metal::detail::GetPreferredNOCForDRAMRead(device->arch());

    // reader kernel
    auto reader_mcast_sender_kernels_id_group_1 = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/dataflow/"
        "reader_mcast_sender_unary_gn.cpp",
        mcast_sender_cores_group_1,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = reader_noc,
            .compile_args = reader_mcast_sender_compile_time_args_group_1,
            .defines = reader_mcast_sender_defines});
    auto reader_mcast_sender_kernels_id_group_2 = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/dataflow/"
        "reader_mcast_sender_unary_gn.cpp",
        mcast_sender_cores_group_2,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = reader_noc,
            .compile_args = reader_mcast_sender_compile_time_args_group_2,
            .defines = reader_mcast_sender_defines});
    KernelHandle reader_mcast_receiver_kernels_id_group_1 = -1;
    KernelHandle reader_mcast_receiver_kernels_id_group_2 = -1;
    if (use_mcast) {
        reader_mcast_receiver_kernels_id_group_1 = CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/dataflow/"
            "reader_mcast_receiver_unary_gn.cpp",
            mcast_receiver_cores_group_1,
            tt::tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                .noc = reader_noc,
                .compile_args = reader_mcast_receiver_compile_time_args_group_1,
                .defines = reader_mcast_receiver_defines});
        reader_mcast_receiver_kernels_id_group_2 = CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/dataflow/"
            "reader_mcast_receiver_unary_gn.cpp",
            mcast_receiver_cores_group_2,
            tt::tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                .noc = reader_noc,
                .compile_args = reader_mcast_receiver_compile_time_args_group_2,
                .defines = reader_mcast_receiver_defines});
    }

    // writer defines
    std::map<string, string> writer_defines;
    // writer compile time args
    std::vector<uint32_t> writer_mcast_sender_compile_time_args_group_1 = {
        1,
        (std::uint32_t)gamma.has_value(),
        (std::uint32_t)beta.has_value(),
        1,
        (std::uint32_t)is_dram(gamma),
        (std::uint32_t)is_dram(beta),
        (std::uint32_t)is_dram(input_mask),
        (std::uint32_t)gamma_beta_num_cols_tile_per_core,
        (std::uint32_t)per_core_Mt_group_1,
        (std::uint32_t)per_core_Nt,
        (std::uint32_t)per_core_N * datum_size_bytes,
        (std::uint32_t)per_core_Nt * TILE_WIDTH * datum_size_bytes,
        (std::uint32_t)num_groups_per_core,
        (std::uint32_t)num_batches_per_core_group_1,
        (std::uint32_t)num_datum_row_per_group_mod_tile_w,
        (std::uint32_t)per_core_Mt_group_1 * Wt / num_batches_per_core_group_1,
        (std::uint32_t)block_wt_last,
        (std::uint32_t)(num_datum_row_per_group_mod_tile_w & (num_datum_row_per_group_mod_tile_w - 1)) == 0,
        (std::uint32_t)num_datum_row_per_group < TILE_WIDTH,
        (std::uint32_t)num_datum_row_per_group - (block_wt - 1) * TILE_WIDTH,
        (std::uint32_t)num_out_blocks,
        (std::uint32_t)block_ht_group_1,
        (std::uint32_t)block_wt,
        (std::uint32_t)block_ht_group_1 * block_wt};
    std::vector<uint32_t> writer_mcast_sender_compile_time_args_group_2 = {
        1,
        (std::uint32_t)gamma.has_value(),
        (std::uint32_t)beta.has_value(),
        1,
        (std::uint32_t)is_dram(gamma),
        (std::uint32_t)is_dram(beta),
        (std::uint32_t)is_dram(input_mask),
        (std::uint32_t)gamma_beta_num_cols_tile_per_core,
        (std::uint32_t)per_core_Mt_group_2,
        (std::uint32_t)per_core_Nt,
        (std::uint32_t)per_core_N * datum_size_bytes,
        (std::uint32_t)per_core_Nt * TILE_WIDTH * datum_size_bytes,
        (std::uint32_t)num_groups_per_core,
        (std::uint32_t)num_batches_per_core_group_2,
        (std::uint32_t)num_datum_row_per_group_mod_tile_w,
        (std::uint32_t)per_core_Mt_group_2 * Wt / num_batches_per_core_group_2,
        (std::uint32_t)block_wt_last,
        (std::uint32_t)(num_datum_row_per_group_mod_tile_w & (num_datum_row_per_group_mod_tile_w - 1)) == 0,
        (std::uint32_t)num_datum_row_per_group < TILE_WIDTH,
        (std::uint32_t)num_datum_row_per_group - (block_wt - 1) * TILE_WIDTH,
        (std::uint32_t)num_out_blocks,
        (std::uint32_t)block_ht_group_2,
        (std::uint32_t)block_wt,
        (std::uint32_t)block_ht_group_2 * block_wt};

    if (gamma.has_value() and gamma.value().layout() == Layout::ROW_MAJOR) {
        auto gamma_stick_size = gamma.value().padded_shape()[3] * gamma.value().element_size();
        bool gamma_stick_size_is_power_of_two = is_power_of_two_at_least_32(gamma_stick_size);
        writer_mcast_sender_compile_time_args_group_1.push_back((std::uint32_t)gamma_stick_size_is_power_of_two);
        writer_mcast_sender_compile_time_args_group_2.push_back((std::uint32_t)gamma_stick_size_is_power_of_two);
        writer_mcast_sender_compile_time_args_group_1.push_back(gamma_stick_size);
        writer_mcast_sender_compile_time_args_group_2.push_back(gamma_stick_size);
    } else if (beta.has_value() and beta.value().layout() == Layout::ROW_MAJOR) {
        auto beta_stick_size = beta.value().padded_shape()[3] * beta.value().element_size();
        bool beta_stick_size_is_power_of_two = is_power_of_two_at_least_32(beta_stick_size);
        writer_mcast_sender_compile_time_args_group_1.push_back((std::uint32_t)beta_stick_size_is_power_of_two);
        writer_mcast_sender_compile_time_args_group_2.push_back((std::uint32_t)beta_stick_size_is_power_of_two);
        writer_mcast_sender_compile_time_args_group_1.push_back(beta_stick_size);
        writer_mcast_sender_compile_time_args_group_2.push_back(beta_stick_size);
    } else {
        writer_mcast_sender_compile_time_args_group_1.push_back(0);
        writer_mcast_sender_compile_time_args_group_2.push_back(0);
    }

    // writer kernel
    bool use_row_major_kernel = true;
    std::string writer_kernel =
        "ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/dataflow/writer_unary_gn_rm_gb.cpp";
    auto writer_kernels_id_group_1 = CreateKernel(
        program,
        writer_kernel,
        all_cores_group_1,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = writer_noc,
            .compile_args = writer_mcast_sender_compile_time_args_group_1,
            .defines = writer_defines});
    auto writer_kernels_id_group_2 = CreateKernel(
        program,
        writer_kernel,
        all_cores_group_2,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = writer_noc,
            .compile_args = writer_mcast_sender_compile_time_args_group_2,
            .defines = writer_defines});
    // defines
    std::map<string, string> eltwise_binary_defines;
    if (reader_repack_output) {
        eltwise_binary_defines["READER_REPACK"] = "1";
    }
    if (tilize_in) {
        eltwise_binary_defines["TILIZE_IN"] = "1";
    }
    if (untilize_out) {
        eltwise_binary_defines["UNTILIZE_OUT"] = "1";
    }
    // compute kernel compile time args
    std::vector<uint32_t> mcast_sender_compute_compile_time_args_group_1 = {
        (std::uint32_t)1,
        (std::uint32_t)gamma.has_value(),
        (std::uint32_t)beta.has_value(),
        (std::uint32_t)num_cores_per_mcast_group,
        (std::uint32_t)num_batches_per_core_group_1,
        (std::uint32_t)num_groups_per_core,

        (std::uint32_t)block_ht_group_1,
        (std::uint32_t)block_wt,
        (std::uint32_t)block_ht_group_1 * block_wt,

        (std::uint32_t)subblock_wt,
        (std::uint32_t)num_subblocks_w,

        (std::uint32_t)per_core_Mt_group_1,
        (std::uint32_t)per_core_Nt,
        (std::uint32_t)per_core_Mt_group_1 * per_core_Nt,

        (std::uint32_t)per_core_Nt * TILE_HW * datum_size_bytes,  // per_core_N_tile_bytes
        (std::uint32_t)num_groups_per_reset,
        (std::uint32_t)single_tile_size,
        (std::uint32_t)per_core_Mt_group_1 * Wt / num_batches_per_core_group_1,
        (std::uint32_t)num_groups_per_core * block_wt,
        (std::uint32_t)num_datum_row_per_group_mod_tile_w,
        (std::uint32_t)block_wt_last,
        (std::uint32_t)(num_datum_row_per_group_mod_tile_w & (num_datum_row_per_group_mod_tile_w - 1)) == 0,
        (std::uint32_t)num_datum_row_per_group < TILE_WIDTH,
        (std::uint32_t)num_datum_row_per_group - (block_wt - 1) * TILE_WIDTH,
        (std::uint32_t)num_out_blocks,
    };
    std::vector<uint32_t> mcast_sender_compute_compile_time_args_group_2 = {
        (std::uint32_t)1,
        (std::uint32_t)gamma.has_value(),
        (std::uint32_t)beta.has_value(),
        (std::uint32_t)num_cores_per_mcast_group,
        (std::uint32_t)num_batches_per_core_group_2,
        (std::uint32_t)num_groups_per_core,

        (std::uint32_t)block_ht_group_2,
        (std::uint32_t)block_wt,
        (std::uint32_t)block_ht_group_2 * block_wt,

        (std::uint32_t)subblock_wt,
        (std::uint32_t)num_subblocks_w,

        (std::uint32_t)per_core_Mt_group_2,
        (std::uint32_t)per_core_Nt,
        (std::uint32_t)per_core_Mt_group_2 * per_core_Nt,

        (std::uint32_t)per_core_Nt * TILE_HW * datum_size_bytes,  // per_core_N_tile_bytes
        (std::uint32_t)num_groups_per_reset,
        (std::uint32_t)single_tile_size,
        (std::uint32_t)per_core_Mt_group_2 * Wt / num_batches_per_core_group_2,
        (std::uint32_t)num_groups_per_core * block_wt,
        (std::uint32_t)num_datum_row_per_group_mod_tile_w,
        (std::uint32_t)block_wt_last,
        (std::uint32_t)(num_datum_row_per_group_mod_tile_w & (num_datum_row_per_group_mod_tile_w - 1)) == 0,
        (std::uint32_t)num_datum_row_per_group < TILE_WIDTH,
        (std::uint32_t)num_datum_row_per_group - (block_wt - 1) * TILE_WIDTH,
        (std::uint32_t)num_out_blocks,
    };

    std::vector<uint32_t> mcast_receiver_compute_compile_time_args_group_1 = {
        (std::uint32_t)0,
        (std::uint32_t)gamma.has_value(),
        (std::uint32_t)beta.has_value(),
        (std::uint32_t)num_cores_per_mcast_group,
        (std::uint32_t)num_batches_per_core_group_1,
        (std::uint32_t)num_groups_per_core,

        (std::uint32_t)block_ht_group_1,
        (std::uint32_t)block_wt,
        (std::uint32_t)block_ht_group_1 * block_wt,

        (std::uint32_t)subblock_wt,
        (std::uint32_t)num_subblocks_w,

        (std::uint32_t)per_core_Mt_group_1,
        (std::uint32_t)per_core_Nt,
        (std::uint32_t)per_core_Mt_group_1 * per_core_Nt,

        (std::uint32_t)per_core_Nt * TILE_HW * datum_size_bytes,  // per_core_N_tile_bytes
        (std::uint32_t)num_groups_per_reset,
        (std::uint32_t)single_tile_size,
        (std::uint32_t)per_core_Mt_group_1 * Wt / num_batches_per_core_group_1,
        (std::uint32_t)num_groups_per_core * block_wt,
        (std::uint32_t)num_datum_row_per_group_mod_tile_w,
        (std::uint32_t)block_wt_last,
        (std::uint32_t)(num_datum_row_per_group_mod_tile_w & (num_datum_row_per_group_mod_tile_w - 1)) == 0,
        (std::uint32_t)num_datum_row_per_group < TILE_WIDTH,
        (std::uint32_t)num_datum_row_per_group - (block_wt - 1) * TILE_WIDTH,
        (std::uint32_t)num_out_blocks,
    };
    std::vector<uint32_t> mcast_receiver_compute_compile_time_args_group_2 = {
        (std::uint32_t)0,
        (std::uint32_t)gamma.has_value(),
        (std::uint32_t)beta.has_value(),
        (std::uint32_t)num_cores_per_mcast_group,
        (std::uint32_t)num_batches_per_core_group_2,
        (std::uint32_t)num_groups_per_core,

        (std::uint32_t)block_ht_group_2,
        (std::uint32_t)block_wt,
        (std::uint32_t)block_ht_group_2 * block_wt,

        (std::uint32_t)subblock_wt,
        (std::uint32_t)num_subblocks_w,

        (std::uint32_t)per_core_Mt_group_2,
        (std::uint32_t)per_core_Nt,
        (std::uint32_t)per_core_Mt_group_2 * per_core_Nt,

        (std::uint32_t)per_core_Nt * TILE_HW * datum_size_bytes,  // per_core_N_tile_bytes
        (std::uint32_t)num_groups_per_reset,
        (std::uint32_t)single_tile_size,
        (std::uint32_t)per_core_Mt_group_2 * Wt / num_batches_per_core_group_2,
        (std::uint32_t)num_groups_per_core * block_wt,
        (std::uint32_t)num_datum_row_per_group_mod_tile_w,
        (std::uint32_t)block_wt_last,
        (std::uint32_t)(num_datum_row_per_group_mod_tile_w & (num_datum_row_per_group_mod_tile_w - 1)) == 0,
        (std::uint32_t)num_datum_row_per_group < TILE_WIDTH,
        (std::uint32_t)num_datum_row_per_group - (block_wt - 1) * TILE_WIDTH,
        (std::uint32_t)num_out_blocks,
    };
    // compute kernel
    bool fp32_dest_acc_en = false;
    bool math_approx_mode = true;
    auto mcast_sender_compute_kernels_id_group_1 = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/compute/groupnorm.cpp",
        mcast_sender_cores_group_1,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .math_approx_mode = math_approx_mode,
            .compile_args = mcast_sender_compute_compile_time_args_group_1,
            .defines = eltwise_binary_defines});
    auto mcast_sender_compute_kernels_id_group_2 = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/compute/groupnorm.cpp",
        mcast_sender_cores_group_2,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .math_approx_mode = math_approx_mode,
            .compile_args = mcast_sender_compute_compile_time_args_group_2,
            .defines = eltwise_binary_defines});
    auto mcast_receiver_compute_kernels_id_group_1 = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/compute/groupnorm.cpp",
        mcast_receiver_cores_group_1,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .math_approx_mode = math_approx_mode,
            .compile_args = mcast_receiver_compute_compile_time_args_group_1,
            .defines = eltwise_binary_defines});
    auto mcast_receiver_compute_kernels_id_group_2 = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/compute/groupnorm.cpp",
        mcast_receiver_cores_group_2,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .math_approx_mode = math_approx_mode,
            .compile_args = mcast_receiver_compute_compile_time_args_group_2,
            .defines = eltwise_binary_defines});

    // Create circular buffers
    uint32_t in0_cb_index = tt::CBIndex::c_0;
    uint32_t output_cb_index = tt::CBIndex::c_16;
    tt::tt_metal::CircularBufferConfig in0_cb_config_group_1 =
        tt::tt_metal::CircularBufferConfig(in0_CB_size_group_1, {{in0_cb_index, in_data_format}})
            .set_page_size(in0_cb_index, in_single_tile_size);
    tt::tt_metal::CircularBufferConfig output_cb_config_group_1 =
        tt::tt_metal::CircularBufferConfig(out_CB_size_group_1, {{output_cb_index, out_data_format}})
            .set_page_size(output_cb_index, out_single_tile_size);

    auto cb_in0_group_1 = tt::tt_metal::CreateCircularBuffer(program, all_cores_group_1, in0_cb_config_group_1);
    auto cb_output_group_1 = tt::tt_metal::CreateCircularBuffer(program, all_cores_group_1, output_cb_config_group_1);
    tt::tt_metal::CircularBufferConfig in0_cb_config_group_2 =
        tt::tt_metal::CircularBufferConfig(in0_CB_size_group_2, {{in0_cb_index, in_data_format}})
            .set_page_size(in0_cb_index, in_single_tile_size);
    tt::tt_metal::CircularBufferConfig output_cb_config_group_2 =
        tt::tt_metal::CircularBufferConfig(out_CB_size_group_2, {{output_cb_index, out_data_format}})
            .set_page_size(output_cb_index, out_single_tile_size);

    auto cb_in0_group_2 = tt::tt_metal::CreateCircularBuffer(program, all_cores_group_2, in0_cb_config_group_2);
    auto cb_output_group_2 = tt::tt_metal::CreateCircularBuffer(program, all_cores_group_2, output_cb_config_group_2);
    // in - stores tilized input
    uint32_t in_cb_index = tt::CBIndex::c_29;
    tt::tt_metal::CircularBufferConfig in_cb_config_group_1 =
        tt::tt_metal::CircularBufferConfig(in_CB_size_group_1, {{in_cb_index, in_data_format}})
            .set_page_size(in_cb_index, in_single_tile_size);
    auto cb_in_group_1 = tt::tt_metal::CreateCircularBuffer(program, all_cores_group_1, in_cb_config_group_1);
    tt::tt_metal::CircularBufferConfig in_cb_config_group_2 =
        tt::tt_metal::CircularBufferConfig(in_CB_size_group_2, {{in_cb_index, in_data_format}})
            .set_page_size(in_cb_index, in_single_tile_size);
    auto cb_in_group_2 = tt::tt_metal::CreateCircularBuffer(program, all_cores_group_2, in_cb_config_group_2);
    // out - stores tilized output
    if (untilize_out) {
        uint32_t out_cb_index = tt::CBIndex::c_30;
        tt::tt_metal::CircularBufferConfig out_cb_config_group_1 =
            tt::tt_metal::CircularBufferConfig(in_CB_size_group_1, {{out_cb_index, in_data_format}})
                .set_page_size(out_cb_index, in_single_tile_size);
        auto cb_out_group_1 = tt::tt_metal::CreateCircularBuffer(program, all_cores_group_1, out_cb_config_group_1);
        tt::tt_metal::CircularBufferConfig out_cb_config_group_2 =
            tt::tt_metal::CircularBufferConfig(in_CB_size_group_2, {{out_cb_index, in_data_format}})
                .set_page_size(out_cb_index, in_single_tile_size);
        auto cb_out_group_2 = tt::tt_metal::CreateCircularBuffer(program, all_cores_group_2, out_cb_config_group_2);
    }
    // in2 scaler - for partial Ex
    uint32_t in2_cb_index = tt::CBIndex::c_2;
    tt::tt_metal::CircularBufferConfig in2_cb_config =
        tt::tt_metal::CircularBufferConfig(in2_CB_size, {{in2_cb_index, cb_data_format}})
            .set_page_size(in2_cb_index, single_tile_size);
    auto cb_in2 = tt::tt_metal::CreateCircularBuffer(program, all_cores, in2_cb_config);
    // in3 eps
    uint32_t in3_cb_index = tt::CBIndex::c_3;
    tt::tt_metal::CircularBufferConfig in3_cb_config =
        tt::tt_metal::CircularBufferConfig(in3_CB_size, {{in3_cb_index, cb_data_format}})
            .set_page_size(in3_cb_index, single_tile_size);
    auto cb_in3 = tt::tt_metal::CreateCircularBuffer(program, all_cores, in3_cb_config);
    // in4 scaler-c
    uint32_t in4_cb_index = tt::CBIndex::c_4;
    tt::tt_metal::CircularBufferConfig in4_cb_config =
        tt::tt_metal::CircularBufferConfig(in2_CB_size, {{in4_cb_index, cb_data_format}})
            .set_page_size(in4_cb_index, single_tile_size);
    auto cb_in4 = tt::tt_metal::CreateCircularBuffer(program, all_cores, in4_cb_config);
    // gamma
    if (gamma.has_value()) {
        uint32_t in5_cb_index = tt::CBIndex::c_5;
        tt::tt_metal::CircularBufferConfig in5_cb_config =
            tt::tt_metal::CircularBufferConfig(in5_CB_size, {{in5_cb_index, gamma_beta_cb_data_format}})
                .set_page_size(in5_cb_index, gamma_beta_single_tile_size);
        auto cb_in5 = tt::tt_metal::CreateCircularBuffer(program, all_cores, in5_cb_config);
    }
    // beta
    if (beta.has_value()) {
        uint32_t in6_cb_index = tt::CBIndex::c_6;
        tt::tt_metal::CircularBufferConfig in6_cb_config =
            tt::tt_metal::CircularBufferConfig(in6_CB_size, {{in6_cb_index, gamma_beta_cb_data_format}})
                .set_page_size(in6_cb_index, gamma_beta_single_tile_size);
        auto cb_in6 = tt::tt_metal::CreateCircularBuffer(program, all_cores, in6_cb_config);
    }
    // input mask
    if (input_mask.has_value()) {
        uint32_t in_mask_cb_index = tt::CBIndex::c_28;
        tt::tt_metal::CircularBufferConfig in_mask_cb_config =
            tt::tt_metal::CircularBufferConfig(in_mask_CB_size, {{in_mask_cb_index, in_mask_cb_data_format}})
                .set_page_size(in_mask_cb_index, in_mask_single_tile_size);
        auto cb_inz = tt::tt_metal::CreateCircularBuffer(program, all_cores, in_mask_cb_config);
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
        auto cb_inz = tt::tt_metal::CreateCircularBuffer(program, all_cores, repack_cb_config);
    }
    // x
    uint32_t x_cb_index = tt::CBIndex::c_24;
    tt::tt_metal::CircularBufferConfig x_cb_config_group_1 =
        tt::tt_metal::CircularBufferConfig(x_CB_size_group_1, {{x_cb_index, cb_data_format}})
            .set_page_size(x_cb_index, single_tile_size);
    auto cb_x_group_1 = tt::tt_metal::CreateCircularBuffer(program, all_cores_group_1, x_cb_config_group_1);
    tt::tt_metal::CircularBufferConfig x_cb_config_group_2 =
        tt::tt_metal::CircularBufferConfig(x_CB_size_group_2, {{x_cb_index, cb_data_format}})
            .set_page_size(x_cb_index, single_tile_size);
    auto cb_x_group_2 = tt::tt_metal::CreateCircularBuffer(program, all_cores_group_2, x_cb_config_group_2);
    // xmm
    uint32_t xmm_cb_index = tt::CBIndex::c_25;
    tt::tt_metal::CircularBufferConfig xmm_cb_config_group_1 =
        tt::tt_metal::CircularBufferConfig(xmm_CB_size_group_1, {{xmm_cb_index, cb_data_format}})
            .set_page_size(xmm_cb_index, single_tile_size);
    auto cb_xmm_group_1 = tt::tt_metal::CreateCircularBuffer(program, all_cores_group_1, xmm_cb_config_group_1);
    tt::tt_metal::CircularBufferConfig xmm_cb_config_group_2 =
        tt::tt_metal::CircularBufferConfig(xmm_CB_size_group_2, {{xmm_cb_index, cb_data_format}})
            .set_page_size(xmm_cb_index, single_tile_size);
    auto cb_xmm_group_2 = tt::tt_metal::CreateCircularBuffer(program, all_cores_group_2, xmm_cb_config_group_2);
    // xmm2
    uint32_t xmm2_cb_index = tt::CBIndex::c_23;
    tt::tt_metal::CircularBufferConfig xmm2_cb_config_group_1 =
        tt::tt_metal::CircularBufferConfig(xmm2_CB_size_group_1, {{xmm2_cb_index, cb_data_format}})
            .set_page_size(xmm2_cb_index, single_tile_size);
    auto cb_xmm2_group_1 = tt::tt_metal::CreateCircularBuffer(program, all_cores_group_1, xmm2_cb_config_group_1);
    tt::tt_metal::CircularBufferConfig xmm2_cb_config_group_2 =
        tt::tt_metal::CircularBufferConfig(xmm2_CB_size_group_2, {{xmm2_cb_index, cb_data_format}})
            .set_page_size(xmm2_cb_index, single_tile_size);
    auto cb_xmm2_group_2 = tt::tt_metal::CreateCircularBuffer(program, all_cores_group_2, xmm2_cb_config_group_2);
    // xmm3
    uint32_t xmm3_cb_index = tt::CBIndex::c_22;
    tt::tt_metal::CircularBufferConfig xmm3_cb_config_group_1 =
        tt::tt_metal::CircularBufferConfig(xmm3_CB_size_group_1, {{xmm3_cb_index, cb_data_format}})
            .set_page_size(xmm3_cb_index, single_tile_size);
    auto cb_xmm3_group_1 = tt::tt_metal::CreateCircularBuffer(program, all_cores_group_1, xmm3_cb_config_group_1);
    tt::tt_metal::CircularBufferConfig xmm3_cb_config_group_2 =
        tt::tt_metal::CircularBufferConfig(xmm3_CB_size_group_2, {{xmm3_cb_index, cb_data_format}})
            .set_page_size(xmm3_cb_index, single_tile_size);
    auto cb_xmm3_group_2 = tt::tt_metal::CreateCircularBuffer(program, all_cores_group_2, xmm3_cb_config_group_2);
    // ex_partial
    uint32_t ex_cb_partial_index = tt::CBIndex::c_8;
    tt::tt_metal::CircularBufferConfig ex_cb_partial_config =
        tt::tt_metal::CircularBufferConfig(ex_partial_CB_size, {{ex_cb_partial_index, cb_data_format}})
            .set_page_size(ex_cb_partial_index, single_tile_size);
    auto cb_ex_partial = tt::tt_metal::CreateCircularBuffer(program, all_cores, ex_cb_partial_config);
    // ex2_partial
    uint32_t ex2_cb_partial_index = tt::CBIndex::c_21;
    tt::tt_metal::CircularBufferConfig ex2_cb_partial_config =
        tt::tt_metal::CircularBufferConfig(ex_partial_CB_size, {{ex2_cb_partial_index, cb_data_format}})
            .set_page_size(ex2_cb_partial_index, single_tile_size);
    auto cb_ex2_partial = tt::tt_metal::CreateCircularBuffer(program, all_cores, ex2_cb_partial_config);
    // ex_external
    uint32_t ex_cb_external_index = tt::CBIndex::c_10;
    tt::tt_metal::CircularBufferConfig ex_cb_external_config =
        tt::tt_metal::CircularBufferConfig(
            2 * single_tile_size * num_cores_per_mcast_group, {{ex_cb_external_index, cb_data_format}})
            .set_page_size(ex_cb_external_index, single_tile_size);
    auto cb_ex_external = tt::tt_metal::CreateCircularBuffer(program, all_cores, ex_cb_external_config);
    // ex_global
    uint32_t ex_cb_index = tt::CBIndex::c_9;
    uint32_t ex_global_cb_index = tt::CBIndex::c_15;
    std::map<uint8_t, tt::DataFormat> ex_global_cb_data_format_spec{
        {ex_global_cb_index, cb_data_format}, {ex_cb_index, cb_data_format}};
    auto ex_global_cb_config = tt::tt_metal::CircularBufferConfig(ex_global_CB_size, ex_global_cb_data_format_spec)
                                   .set_page_size(ex_global_cb_index, single_tile_size)
                                   .set_page_size(ex_cb_index, single_tile_size);
    auto cb_ex_global = tt::tt_metal::CreateCircularBuffer(program, all_cores, ex_global_cb_config);
    // ex2_global
    uint32_t ex2_cb_index = tt::CBIndex::c_13;
    uint32_t ex2_global_cb_index = tt::CBIndex::c_14;
    std::map<uint8_t, tt::DataFormat> ex2_global_cb_data_format_spec{
        {ex2_global_cb_index, cb_data_format}, {ex2_cb_index, cb_data_format}};
    auto ex2_global_cb_config = tt::tt_metal::CircularBufferConfig(ex2_global_CB_size, ex2_global_cb_data_format_spec)
                                    .set_page_size(ex2_global_cb_index, single_tile_size)
                                    .set_page_size(ex2_cb_index, single_tile_size);
    auto cb2_ex_global = tt::tt_metal::CreateCircularBuffer(program, all_cores, ex2_global_cb_config);
    // ex2pe
    uint32_t cb_ex2pe_index;
    cb_ex2pe_index = tt::CBIndex::c_27;
    tt::tt_metal::CircularBufferConfig ex2pe_cb_config =
        tt::tt_metal::CircularBufferConfig(ex2pe_CB_size, {{cb_ex2pe_index, cb_data_format}})
            .set_page_size(cb_ex2pe_index, single_tile_size);
    auto cb_ex2pe = tt::tt_metal::CreateCircularBuffer(program, all_cores, ex2pe_cb_config);

    // Runtime Args
    std::vector<KernelHandle> writer_kernel_ids;
    std::vector<KernelHandle> reader_sender_kernel_ids;
    std::vector<KernelHandle> reader_receiver_kernel_ids;
    float winv_group_1 =
        1.0f / std::sqrt(num_rows_per_batch_per_core_group_1 * num_datum_row_per_group);  // bcast-w scaler
    bfloat16 bfloat_winv_value_group_1 = bfloat16(winv_group_1);
    uint32_t packed_winv_value_group_1 =
        pack_two_bfloat16_into_uint32({bfloat_winv_value_group_1, bfloat_winv_value_group_1});
    float winv_group_2 = winv_group_1;
    bfloat16 bfloat_winv_value_group_2 = bfloat_winv_value_group_1;
    uint32_t packed_winv_value_group_2 = packed_winv_value_group_1;
    if (num_batches_per_core_group_2 > 0) {
        winv_group_2 =
            1.0f / std::sqrt(num_rows_per_batch_per_core_group_2 * num_datum_row_per_group);  // bcast-w scaler
        bfloat_winv_value_group_2 = bfloat16(winv_group_2);
        packed_winv_value_group_2 =
            pack_two_bfloat16_into_uint32({bfloat_winv_value_group_2, bfloat_winv_value_group_2});
    }
    float cinv = 1.0f / std::sqrt(num_cores_per_batch * num_cores_per_group);  // bcast-cores scaler
    bfloat16 bfloat_cinv_value = bfloat16(cinv);
    uint32_t packed_cinv_value = pack_two_bfloat16_into_uint32({bfloat_cinv_value, bfloat_cinv_value});
    union {
        float f;
        uint32_t u;
    } e;
    e.f = eps;

    log_debug(tt::LogOp, "num_rows_per_batch_per_core_group_1: {}", num_rows_per_batch_per_core_group_1);
    log_debug(tt::LogOp, "num_rows_per_batch_per_core_group_2: {}", num_rows_per_batch_per_core_group_2);
    log_debug(tt::LogOp, "num_datum_row_per_group: {}", num_datum_row_per_group);
    log_debug(tt::LogOp, "num_cores_per_batch: {}", num_cores_per_batch);
    log_debug(tt::LogOp, "num_cores_per_group: {}", num_cores_per_group);

    for (int i = 0; i < mcast_groups.size(); ++i) {
        auto group = mcast_groups[i];
        bool rectangle_grid = is_rectangle_grid(group);

        for (int j = 0; j < group.size(); ++j) {
            CoreCoord core = group[j];
            CoreCoord core_physical = device->worker_core_from_logical_core(core);
            uint32_t in0_start_id, out_tile_start_id;
            if (equal_batches_per_core || (core.x <= last_row_with_extra_batch)) {
                in0_start_id = per_core_Mt_group_1 * Wt * core.x + per_core_Nt * core.y;
                out_tile_start_id = per_core_Mt_group_1 * Wt * core.x + per_core_Nt * core.y;
            } else {
                in0_start_id = per_core_Mt_group_1 * Wt * (last_row_with_extra_batch + 1) +
                               per_core_Mt_group_2 * Wt * (core.x - last_row_with_extra_batch - 1) +
                               per_core_Nt * core.y;
                out_tile_start_id = in0_start_id;
            }

            if (j == 0) {  // mcast sender
                // get the bounding box for the mcast
                std::vector<CoreCoord> mcast_group_first;
                std::vector<CoreCoord> mcast_group_mid(group);
                std::vector<CoreCoord> mcast_group_last;
                if (not rectangle_grid) {
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
                mcast_sender_args.push_back(not mcast_group_first.empty());
                mcast_sender_args.push_back(not mcast_group_last.empty());
                mcast_sender_args.push_back(mcast_start.x);
                mcast_sender_args.push_back(mcast_start.y);
                mcast_sender_args.push_back(mcast_end.x);
                mcast_sender_args.push_back(mcast_end.y);
                if (not mcast_group_first.empty()) {
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

                if (not mcast_group_first.empty()) {
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
                if (not mcast_group_last.empty()) {
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
                for (int c = 0; c < group.size(); ++c) {
                    CoreCoord coord = device->worker_core_from_logical_core(group[c]);
                    mcast_noc_xy.push_back(coord.x);
                }
                for (int c = 0; c < group.size(); ++c) {
                    CoreCoord coord = device->worker_core_from_logical_core(group[c]);
                    mcast_noc_xy.push_back(coord.y);
                }
                mcast_sender_args.insert(mcast_sender_args.end(), mcast_noc_xy.begin(), mcast_noc_xy.end());
                if (equal_batches_per_core || (core.x <= last_row_with_extra_batch)) {
                    tt::tt_metal::SetRuntimeArgs(
                        program, reader_mcast_sender_kernels_id_group_1, core, mcast_sender_args);
                    reader_sender_kernel_ids.push_back(reader_mcast_sender_kernels_id_group_1);
                } else {
                    tt::tt_metal::SetRuntimeArgs(
                        program, reader_mcast_sender_kernels_id_group_2, core, mcast_sender_args);
                    reader_sender_kernel_ids.push_back(reader_mcast_sender_kernels_id_group_2);
                }
            } else {  // mcast receiver
                log_debug(tt::LogOp, "mcast receiver receive from coord: {} {}", group.front().x, group.front().y);

                std::vector<uint32_t> mcast_receiver_args = {
                    (std::uint32_t)in0_dram_addr,      // in0_tensor_addr
                    (std::uint32_t)out_dram_addr,      // out_dram_addr
                    (std::uint32_t)in0_start_id,       // in0_tensor_start_tile_id
                    (std::uint32_t)out_tile_start_id,  // out_tensor_start_tile_id
                    (std::uint32_t)Wt,                 // num channel tiles
                    (std::uint32_t)(device->worker_core_from_logical_core(group.front()).x),
                    (std::uint32_t)(device->worker_core_from_logical_core(group.front()).y)};
                if (equal_batches_per_core || (core.x <= last_row_with_extra_batch)) {
                    tt::tt_metal::SetRuntimeArgs(
                        program, reader_mcast_receiver_kernels_id_group_1, core, mcast_receiver_args);
                    reader_receiver_kernel_ids.push_back(reader_mcast_receiver_kernels_id_group_1);
                } else {
                    tt::tt_metal::SetRuntimeArgs(
                        program, reader_mcast_receiver_kernels_id_group_2, core, mcast_receiver_args);
                    reader_receiver_kernel_ids.push_back(reader_mcast_receiver_kernels_id_group_2);
                }
            }
        }
    }

    // writer
    uint32_t gamma_tile_start_id = 0;
    uint32_t beta_tile_start_id = 0;
    uint32_t input_mask_tile_start_id = 0;
    for (int i = 0; i < core_coords.size(); ++i) {
        auto core = core_coords[i];
        uint32_t out_tile_start_id;
        if (equal_batches_per_core || (core.x <= last_row_with_extra_batch)) {
            out_tile_start_id = per_core_Mt_group_1 * Wt * core.x + per_core_Nt * core.y;
        } else {
            out_tile_start_id = per_core_Mt_group_1 * Wt * (last_row_with_extra_batch + 1) +
                                per_core_Mt_group_2 * Wt * (core.x - last_row_with_extra_batch - 1) +
                                per_core_Nt * core.y;
        }

        std::vector<uint32_t> writer_mcast_sender_args;
        writer_mcast_sender_args.push_back(packed_cinv_value);
        if (equal_batches_per_core || (core.x <= last_row_with_extra_batch)) {
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
        if (equal_batches_per_core || (core.x <= last_row_with_extra_batch)) {
            tt::tt_metal::SetRuntimeArgs(program, writer_kernels_id_group_1, core, writer_mcast_sender_args);
            writer_kernel_ids.push_back(writer_kernels_id_group_1);
        } else {
            tt::tt_metal::SetRuntimeArgs(program, writer_kernels_id_group_2, core, writer_mcast_sender_args);
            writer_kernel_ids.push_back(writer_kernels_id_group_2);
        }

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
    auto override_runtime_args_callback =
        [writer_kernel_ids, reader_sender_kernel_ids, reader_receiver_kernel_ids, num_cores, grid_size, mcast_groups](
            const void* operation,
            Program& program,
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<Tensor>& output_tensors) {
            auto src_buffer_a = input_tensors.at(0).buffer()->address();
            auto gamma_tensor = optional_input_tensors.at(0);
            auto beta_tensor = optional_input_tensors.at(1);
            auto mask_tensor = optional_input_tensors.at(2);
            auto dst_buffer = output_tensors.at(0).buffer()->address();

            // updatedynamiccircularbufferaddress(program, cb_in0, *src_buffer_a);
            // updatedynamiccircularbufferaddress(program, cb_output, *dst_buffer);
            for (uint32_t i = 0; i < num_cores; ++i) {
                CoreCoord core = {i % grid_size.x, i / grid_size.x};

                auto writer_kernel_id = writer_kernel_ids.at(i);
                auto& writer_runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);

                writer_runtime_args[3] = dst_buffer;
                if (gamma_tensor.has_value()) {
                    writer_runtime_args[4] = gamma_tensor.value().buffer()->address();
                }
                if (beta_tensor.has_value()) {
                    writer_runtime_args[5] = beta_tensor.value().buffer()->address();
                }
                if (mask_tensor.has_value()) {
                    writer_runtime_args[6] = mask_tensor.value().buffer()->address();
                }
            }
            uint32_t sender_index = 0;
            uint32_t receiver_index = 0;
            for (int i = 0; i < mcast_groups.size(); ++i) {
                auto group = mcast_groups[i];
                for (int j = 0; j < group.size(); ++j) {
                    CoreCoord core = group[j];
                    if (j == 0) {
                        auto reader_sender_kernel_id = reader_sender_kernel_ids.at(sender_index);
                        auto& reader_sender_runtime_args = GetRuntimeArgs(program, reader_sender_kernel_id, core);
                        reader_sender_runtime_args[0] = src_buffer_a;
                        reader_sender_runtime_args[1] = dst_buffer;
                        sender_index++;
                    } else {
                        auto reader_receiver_kernel_id = reader_receiver_kernel_ids.at(receiver_index);
                        auto& reader_receiver_runtime_args = GetRuntimeArgs(program, reader_receiver_kernel_id, core);
                        reader_receiver_runtime_args[0] = src_buffer_a;
                        reader_receiver_runtime_args[1] = dst_buffer;
                        receiver_index++;
                    }
                }
            }
        };
    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_args_callback};
}

}  // namespace ttnn::operations::normalization

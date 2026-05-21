// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"

namespace ttnn::prim::sharded_layernorm_helpers {

using namespace tt::tt_metal;

// Forward declarations
struct GridParams;
struct WorkerDistribution;
struct CoreRanges;

//////////////////////////////////////////////////////////////////////////////
// Validation and data format helpers
//////////////////////////////////////////////////////////////////////////////

void assert_subblock_compute_config_compatible(bool dst_full_sync_en, bool fp32_dest_acc_en, uint32_t subblock_wt);

std::tuple<tt::DataFormat, tt::DataFormat, tt::DataFormat, tt::DataFormat, tt::DataFormat, tt::DataFormat>
get_cb_data_formats(
    const Tensor& output,
    const std::optional<const Tensor>& gamma,
    const std::optional<const Tensor>& beta,
    const std::optional<const Tensor>& stats,
    bool fp32_dest_acc_en);

//////////////////////////////////////////////////////////////////////////////
// Grid and worker distribution structs (pure logic; reused across legacy & Metal 2.0 ports)
//////////////////////////////////////////////////////////////////////////////

struct GridParams {
    ShardSpec shard_spec;
    CoreCoord grid_size;
    std::optional<CoreCoord> grid_offset;
    bool mcast_1d = false;
    bool row_wise = false;
    uint32_t num_blocks = 0;
    bool use_mcast = false;
    bool use_two_stage_reduce = false;

    static GridParams compute(const Tensor& input, uint32_t block_ht, CoreCoord compute_with_storage_grid_size);
};

struct WorkerDistribution {
    uint32_t num_rows_per_all_to_all_worker = 0;
    uint32_t num_rows_per_all_to_all_worker_last = 0;
    uint32_t num_cores_all_to_all = 0;
    uint32_t num_cores_all_to_all_first_stage = 0;
    uint32_t num_cores_all_to_all_second_stage = 0;
    uint32_t num_none_all_to_all_workers = 0;
    uint32_t num_blocks_first_stage = 0;
    uint32_t num_blocks_second_stage = 0;

    static WorkerDistribution compute(const GridParams& grid, uint32_t block_ht);
};

struct CoreRanges {
    CoreCoord start_core;
    CoreRangeSet all_cores;
    CoreRange sender_cores{{0, 0}, {0, 0}};
    CoreRangeSet all_to_all_cores;
    CoreRangeSet all_to_all_workers_except_sender;
    CoreRangeSet not_all_to_all_workers;
    uint32_t num_cores_x_mcast = 0;
    uint32_t num_cores_y_mcast = 0;

    static CoreRanges compute(const GridParams& grid, const WorkerDistribution& workers);
};

//////////////////////////////////////////////////////////////////////////////
// Kernel paths
//////////////////////////////////////////////////////////////////////////////

struct KernelPaths {
    std::string reader_sender;
    std::string reader_receiver;
    std::string writer;
    std::string compute;

    static KernelPaths get(
        bool is_pre_all_gather, bool is_post_all_gather, bool use_row_major_kernel, bool use_welford);
};

//////////////////////////////////////////////////////////////////////////////
// Defines (kernel preprocessor symbols). Defines is a vector<pair<string,string>>,
// compatible with both legacy KernelDescriptor::Defines and Metal 2.0
// KernelSpec::CompilerOptions::Defines.
//////////////////////////////////////////////////////////////////////////////

using Defines = std::vector<std::pair<std::string, std::string>>;

struct KernelDefines {
    Defines reader;
    Defines writer;
    Defines compute;

    static KernelDefines build(
        bool has_b,
        bool has_gamma,
        bool has_beta,
        bool rms_norm,
        bool use_welford,
        bool skip_write_back,
        const std::optional<operations::unary::UnaryWithParam>& fused_activation = std::nullopt,
        std::optional<tt::tt_metal::DataType> output_dtype = std::nullopt);
};

//////////////////////////////////////////////////////////////////////////////
// CB sizing
//////////////////////////////////////////////////////////////////////////////

struct CBSizeParams {
    uint32_t block_ht = 0;
    uint32_t block_wt = 0;
    uint32_t block_wt_resharded = 0;
    uint32_t Kt = 0;
    uint32_t in_single_tile_size = 0;
    uint32_t single_tile_size = 0;
    uint32_t out_single_tile_size = 0;
    uint32_t gamma_single_tile_size = 0;
    uint32_t beta_single_tile_size = 0;
    uint32_t stats_single_tile_size = 0;
    uint32_t bfloat16_tile_size = 0;
    uint32_t reciprocal_CB_size_bytes = 0;
    uint32_t num_rows_per_all_to_all_worker = 0;
    uint32_t num_blocks_first_stage = 0;
    uint32_t num_blocks_second_stage = 0;
    uint32_t pre_all_gather_stats_block_tiles = 0;
    uint32_t post_all_gather_stats_block_tiles = 0;
    bool is_pre_all_gather = false;
    bool is_post_all_gather = false;
    bool use_two_stage_reduce = false;
    bool use_welford = false;
    bool skip_write_back = false;
    bool rms_norm = false;

    struct Sizes {
        uint32_t in0_CB_size = 0;
        uint32_t in1_CB_size = 0;
        uint32_t in2_CB_size = 0;
        uint32_t in3_CB_size = 0;
        uint32_t in5_CB_size = 0;
        uint32_t in6_CB_size = 0;
        uint32_t x_CB_size = 0;
        uint32_t xmm_CB_size = 0;
        uint32_t ex_partial_CB_size = 0;
        uint32_t ex_CB_size = 0;
        uint32_t ex_external_CB_size = 0;
        uint32_t ex_global_CB_size = 0;
        uint32_t ex2pe_CB_size = 0;
        uint32_t out_CB_size = 0;
        uint32_t out_reshard_CB_size = 0;
        uint32_t stats_cb_size = 0;
        uint32_t stats_reduced_cb_size = 0;
    };

    Sizes compute() const;
};

//////////////////////////////////////////////////////////////////////////////
// Per-core indices (used for runtime arg construction)
//////////////////////////////////////////////////////////////////////////////

struct PerCoreIndices {
    uint32_t height_index = 0;
    uint32_t width_index = 0;
    uint32_t width_index_two_stage = 0;
    uint32_t all_to_all_worker_tile_offset_bytes = 0;
    uint32_t gamma_tile_start_id = 0;
    uint32_t beta_tile_start_id = 0;
    uint32_t num_reduce_tiles_per_block_h = 0;

    static PerCoreIndices compute(
        uint32_t core_idx,
        const CoreCoord& core,
        const GridParams& grid,
        const WorkerDistribution& workers,
        uint32_t block_wt,
        uint32_t Kt,
        uint32_t last_core_width_index,
        uint32_t single_tile_size);

    bool is_all_to_all(const GridParams& grid, const WorkerDistribution& workers) const;
};

}  // namespace ttnn::prim::sharded_layernorm_helpers

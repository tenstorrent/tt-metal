// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>

#include "ttnn/tensor/tensor.hpp"

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

std::tuple<tt::DataFormat, tt::DataFormat, tt::DataFormat, tt::DataFormat, tt::DataFormat> get_cb_data_formats(
    const Tensor& output,
    const std::optional<const Tensor>& gamma,
    const std::optional<const Tensor>& beta,
    bool fp32_dest_acc_en);

//////////////////////////////////////////////////////////////////////////////
// Grid and worker distribution structs
//////////////////////////////////////////////////////////////////////////////

// Struct to hold grid parameters computed from tensor shard spec
struct GridParams {
    ShardSpec shard_spec;
    CoreCoord grid_size{};
    std::optional<CoreCoord> grid_offset{};
    bool mcast_1d = false;
    bool row_wise = false;
    uint32_t num_blocks = 0;
    bool use_mcast = false;
    bool use_two_stage_reduce = false;

    static GridParams compute(const Tensor& input, uint32_t block_ht, CoreCoord compute_with_storage_grid_size);
};

// Struct to hold worker distribution parameters
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

// Struct to hold computed core ranges for kernels
struct CoreRanges {
    CoreCoord start_core{};
    CoreRangeSet all_cores{};
    CoreRange sender_cores{{0, 0}, {0, 0}};
    CoreRangeSet all_to_all_cores{};
    CoreRangeSet all_to_all_workers_except_sender{};
    CoreRangeSet not_all_to_all_workers{};
    uint32_t num_cores_x_mcast = 0;
    uint32_t num_cores_y_mcast = 0;

    static CoreRanges compute(const GridParams& grid, const WorkerDistribution& workers);
};

//////////////////////////////////////////////////////////////////////////////
// Kernel paths, defines, and compile-time args helpers
//////////////////////////////////////////////////////////////////////////////

// Struct to hold kernel file paths based on operation mode
struct KernelPaths {
    std::string reader_sender;
    std::string reader_receiver;
    std::string writer;
    std::string compute;

    static KernelPaths get(
        bool is_pre_all_gather, bool is_post_all_gather, bool use_row_major_kernel, bool use_welford);
};

// Struct to hold kernel defines for all kernel types
struct KernelDefines {
    KernelDescriptor::Defines reader;
    KernelDescriptor::Defines writer;
    KernelDescriptor::Defines compute;

    static KernelDefines build(
        bool has_b, bool has_gamma, bool has_beta, bool rms_norm, bool use_welford, bool skip_write_back);
};

// Parameters needed to compute CB sizes
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

    // Computes all CB sizes and returns them in a struct
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

// Context needed to build compile-time args for all kernels
struct CompileTimeArgsContext {
    // Semaphore IDs
    uint32_t reduce_receiver_semaphore_id = 0;
    uint32_t reduce_sender_semaphore_id = 0;
    uint32_t reduce_second_stage_semaphore_id = 0;

    // Grid and worker params
    const GridParams* grid = nullptr;
    const WorkerDistribution* workers = nullptr;
    const CoreRanges* core_ranges = nullptr;

    // Block dimensions
    uint32_t block_ht = 0;
    uint32_t block_wt = 0;
    uint32_t subblock_wt = 0;
    uint32_t single_tile_size = 0;
    uint32_t out_single_tile_size = 0;
    uint32_t block_wt_resharded = 0;
    uint32_t K = 0;

    // Flags
    bool rms_norm = false;
    bool use_welford = false;
    bool has_gamma = false;
    bool has_beta = false;
    bool fp32_dest_acc_en = false;
    bool legacy_reduction = false;
    bool legacy_rsqrt = false;

    // Data formats
    tt::DataFormat gamma_cb_data_format = tt::DataFormat::Float16_b;
    tt::DataFormat beta_cb_data_format = tt::DataFormat::Float16_b;

    // Tensor buffers for TensorAccessorArgs
    Buffer* gamma_buffer = nullptr;
    Buffer* beta_buffer = nullptr;

    // For row-major gamma/beta
    bool gamma_is_row_major = false;
    bool beta_is_row_major = false;
    uint32_t gamma_stick_size = 0;
    uint32_t beta_stick_size = 0;

    // Welford-specific
    float eps = 0.0f;
    uint32_t per_core_recip_lut_size = 0;
};

// Result of building compile-time args
struct CompileTimeArgs {
    std::vector<uint32_t> reader_sender;
    std::vector<uint32_t> reader_receiver_all_to_all;
    std::vector<uint32_t> reader_receiver;
    std::vector<uint32_t> writer_sender;
    std::vector<uint32_t> writer_receiver;
    std::vector<uint32_t> compute_all_to_all;
    std::vector<uint32_t> compute_not_all_to_all;

    static CompileTimeArgs build(const CompileTimeArgsContext& ctx);
};

//////////////////////////////////////////////////////////////////////////////
// Kernel and CB configuration structs
//////////////////////////////////////////////////////////////////////////////

// Struct to hold kernel configuration for building kernel descriptors
struct KernelConfig {
    // Paths
    std::string reader_sender_path;
    std::string reader_receiver_path;
    std::string writer_path;
    std::string compute_path;

    // Compile time args
    std::vector<uint32_t> reader_sender_ct_args;
    std::vector<uint32_t> reader_receiver_all_to_all_ct_args;
    std::vector<uint32_t> reader_receiver_ct_args;
    std::vector<uint32_t> writer_sender_ct_args;
    std::vector<uint32_t> writer_receiver_ct_args;
    std::vector<uint32_t> compute_all_to_all_ct_args;
    std::vector<uint32_t> compute_not_all_to_all_ct_args;

    // Defines
    KernelDescriptor::Defines reader_sender_defines;
    KernelDescriptor::Defines reader_receiver_defines;
    KernelDescriptor::Defines writer_defines;
    KernelDescriptor::Defines compute_defines;

    // Runtime args
    KernelDescriptor::RuntimeArgs reader_sender_rt_args;
    KernelDescriptor::RuntimeArgs reader_receiver_all_to_all_rt_args;
    KernelDescriptor::RuntimeArgs reader_receiver_rt_args;
    KernelDescriptor::RuntimeArgs writer_sender_rt_args;
    KernelDescriptor::RuntimeArgs writer_receiver_rt_args;
    KernelDescriptor::RuntimeArgs compute_all_to_all_rt_args;
    KernelDescriptor::RuntimeArgs compute_not_all_to_all_rt_args;

    // NOC config
    tt::tt_metal::NOC reader_noc = tt::tt_metal::NOC::NOC_0;
    tt::tt_metal::NOC writer_noc = tt::tt_metal::NOC::NOC_0;

    // Compute config
    MathFidelity math_fidelity = MathFidelity::HiFi4;
    bool fp32_dest_acc_en = false;
    bool math_approx_mode = false;
};

// Struct to hold CB configuration for building CB descriptors
struct CBConfig {
    // Sizes
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
    uint32_t reciprocal_CB_size_bytes = 0;

    // Data formats
    tt::DataFormat in_data_format = tt::DataFormat::Float16_b;
    tt::DataFormat cb_data_format = tt::DataFormat::Float16_b;
    tt::DataFormat out_data_format = tt::DataFormat::Float16_b;
    tt::DataFormat gamma_cb_data_format = tt::DataFormat::Float16_b;
    tt::DataFormat beta_cb_data_format = tt::DataFormat::Float16_b;
    tt::DataFormat reciprocal_cb_data_format = tt::DataFormat::Float32;

    // Tile sizes
    uint32_t in_single_tile_size = 0;
    uint32_t single_tile_size = 0;
    uint32_t out_single_tile_size = 0;
    uint32_t gamma_single_tile_size = 0;
    uint32_t beta_single_tile_size = 0;
    uint32_t bfloat16_tile_size = 0;

    // Buffers
    Buffer* a_buffer = nullptr;
    Buffer* b_buffer = nullptr;
    Buffer* gamma_buffer = nullptr;
    Buffer* beta_buffer = nullptr;
    Buffer* stats_buffer = nullptr;
    Buffer* recip_buffer = nullptr;
    Buffer* output_buffer = nullptr;

    // Flags
    bool has_b = false;
    bool has_gamma = false;
    bool has_beta = false;
    bool rms_norm = false;
    bool use_welford = false;
    bool is_pre_all_gather = false;
    bool is_post_all_gather = false;
    bool skip_write_back = false;
};

//////////////////////////////////////////////////////////////////////////////
// Kernel and CB descriptor builders
//////////////////////////////////////////////////////////////////////////////

void add_kernel_descriptors(
    ProgramDescriptor& program_descriptor,
    const CoreRanges& core_ranges,
    const WorkerDistribution& workers,
    const GridParams& grid,
    KernelConfig&& kernel_config);

void add_cb_descriptors(
    ProgramDescriptor& program_descriptor,
    const CoreRanges& core_ranges,
    const CoreRangeSet& all_worker_and_storage_cores,
    const CBConfig& cb_config);

//////////////////////////////////////////////////////////////////////////////
// Runtime args building
//////////////////////////////////////////////////////////////////////////////

// Struct to hold context for building runtime args
struct RuntimeArgsContext {
    // Grid and worker info
    const GridParams& grid;
    const WorkerDistribution& workers;
    const CoreRanges& core_ranges;

    // NOC coordinates for multicast
    std::vector<uint32_t> mcast_noc_x;
    std::vector<uint32_t> mcast_noc_y;

    // Packed values for writer
    uint32_t packed_cinv_value = 0;
    uint32_t packed_cinv_value_one = 0;
    uint32_t packed_winv_value = 0;
    uint32_t eps_u = 0;

    // Addresses
    uint32_t gamma_dram_addr = 0;
    uint32_t beta_dram_addr = 0;

    // Tile and block info
    uint32_t single_tile_size = 0;
    uint32_t out_single_tile_size = 0;
    uint32_t block_wt = 0;
    uint32_t block_wt_resharded = 0;
    uint32_t Kt = 0;
    uint32_t last_core_width_index = 0;

    // Flags
    bool is_post_all_gather = false;
    uint32_t num_distributed_devices = 1;
    tt::tt_metal::NOC reader_noc = tt::tt_metal::NOC::NOC_0;

    // Storage core info for write-back
    std::vector<uint32_t> storage_core_noc_x;
    std::vector<uint32_t> storage_core_noc_y;
    uint32_t num_storage_cores = 0;
};

// Per-core indices computed from core position
struct CoreIndices {
    uint32_t height_index = 0;
    uint32_t width_index = 0;
    uint32_t width_index_two_stage = 0;
    uint32_t all_to_all_worker_tile_offset_bytes = 0;
    uint32_t gamma_tile_start_id = 0;
    uint32_t beta_tile_start_id = 0;
    uint32_t num_reduce_tiles_per_block_h = 0;

    static CoreIndices compute(uint32_t core_idx, const CoreCoord& core, const RuntimeArgsContext& ctx);

    // Returns true if this core is an all-to-all worker based on its indices
    bool is_all_to_all(const RuntimeArgsContext& ctx) const;
};

// Struct to hold all runtime args output
struct RuntimeArgsResult {
    KernelDescriptor::RuntimeArgs reader_sender;
    KernelDescriptor::RuntimeArgs reader_receiver_all_to_all;
    KernelDescriptor::RuntimeArgs reader_receiver;
    KernelDescriptor::RuntimeArgs writer_sender;
    KernelDescriptor::RuntimeArgs writer_receiver;
    KernelDescriptor::RuntimeArgs compute_all_to_all;
    KernelDescriptor::RuntimeArgs compute_not_all_to_all;

    static RuntimeArgsResult build(const std::vector<CoreCoord>& cores, RuntimeArgsContext& ctx, IDevice* device);
};

}  // namespace ttnn::prim::sharded_layernorm_helpers

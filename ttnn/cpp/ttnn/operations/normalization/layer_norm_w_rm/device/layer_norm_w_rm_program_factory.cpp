// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "layer_norm_w_rm_program_factory.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operations/cb_utils.hpp"

namespace ttnn::operations::normalization::layer_norm_w_rm::program {

using namespace tt;
using namespace tt::tt_metal;

LayerNormWRmProgramFactory::cached_program_t LayerNormWRmProgramFactory::create(
    const LayerNormWRmParams& operation_attributes,
    const LayerNormWRmInputs& tensor_args,
    Tensor& tensor_return_value) {
    // ============================================================
    // CONSTANTS
    // ============================================================
    constexpr uint32_t buffering_factor = 2;  // Double buffering

    // ============================================================
    // 1. Extract tensor properties
    // ============================================================
    const auto& input = tensor_args.input;
    const auto& gamma = tensor_args.gamma;
    const auto& beta = tensor_args.beta;
    const auto& output = tensor_return_value;

    const auto* src_buffer = input.buffer();
    const auto* gamma_buffer = gamma.buffer();
    const auto* beta_buffer = beta.buffer();
    const auto* dst_buffer = output.buffer();

    const tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    const uint32_t tile_size = tt::tile_size(cb_data_format);

    // Tensor dimensions (ROW_MAJOR layout)
    const uint32_t W = input.padded_shape()[-1];         // Width in elements (padded)
    const uint32_t H = input.padded_shape()[-2];         // Height in elements (padded)
    const uint32_t Wt = W / tt::constants::TILE_WIDTH;   // Width in tiles
    const uint32_t Ht = H / tt::constants::TILE_HEIGHT;  // Height in tiles

    // Stick sizes (ROW_MAJOR)
    const uint32_t input_stick_size = W * input.element_size();
    const uint32_t input_stick_size_aligned = tt::round_up(input_stick_size, src_buffer->alignment());
    const uint32_t gamma_stick_size = W * gamma.element_size();
    const uint32_t gamma_stick_size_aligned = tt::round_up(gamma_stick_size, gamma_buffer->alignment());
    const uint32_t beta_stick_size = W * beta.element_size();
    const uint32_t beta_stick_size_aligned = tt::round_up(beta_stick_size, beta_buffer->alignment());
    const uint32_t output_stick_size = W * output.element_size();
    const uint32_t output_stick_size_aligned = tt::round_up(output_stick_size, dst_buffer->alignment());

    // ============================================================
    // 2. Create program
    // ============================================================
    Program program = Program();

    // ============================================================
    // 3. Work distribution (single core for now)
    // ============================================================
    const CoreCoord core = {0, 0};
    const CoreRange all_cores(core, core);
    const CoreRangeSet core_range_set({all_cores});

    // ============================================================
    // 4. Create circular buffers (15 CBs total)
    // ============================================================

    // CB c_0: Input RM sticks (double-buffered)
    constexpr uint32_t cb_in_rm_idx = tt::CBIndex::c_0;
    const uint32_t cb_in_rm_page_size = tile_size;
    const uint32_t cb_in_rm_num_pages = buffering_factor * Wt;
    tt::tt_metal::create_cb(
        cb_in_rm_idx, program, core_range_set, cb_in_rm_page_size, cb_in_rm_num_pages, cb_data_format);

    // CB c_1: Tiled input (PERSISTENT for reduce + subtract)
    constexpr uint32_t cb_in_tiled_idx = tt::CBIndex::c_1;
    const uint32_t cb_in_tiled_page_size = tile_size;
    const uint32_t cb_in_tiled_num_pages = Wt;
    tt::tt_metal::create_cb(
        cb_in_tiled_idx, program, core_range_set, cb_in_tiled_page_size, cb_in_tiled_num_pages, cb_data_format);

    // CB c_2: Scaler (1/W) for reduces
    constexpr uint32_t cb_scaler_idx = tt::CBIndex::c_2;
    const uint32_t cb_scaler_page_size = tile_size;
    const uint32_t cb_scaler_num_pages = 1;
    tt::tt_metal::create_cb(
        cb_scaler_idx, program, core_range_set, cb_scaler_page_size, cb_scaler_num_pages, cb_data_format);

    // CB c_3: Mean tile
    constexpr uint32_t cb_mean_idx = tt::CBIndex::c_3;
    const uint32_t cb_mean_page_size = tile_size;
    const uint32_t cb_mean_num_pages = 1;
    tt::tt_metal::create_cb(cb_mean_idx, program, core_range_set, cb_mean_page_size, cb_mean_num_pages, cb_data_format);

    // CB c_4: Centralized tiles (PERSISTENT through phases 3-8)
    constexpr uint32_t cb_centralized_idx = tt::CBIndex::c_4;
    const uint32_t cb_centralized_page_size = tile_size;
    const uint32_t cb_centralized_num_pages = Wt;
    tt::tt_metal::create_cb(
        cb_centralized_idx,
        program,
        core_range_set,
        cb_centralized_page_size,
        cb_centralized_num_pages,
        cb_data_format);

    // CB c_5: Squared tiles
    constexpr uint32_t cb_squared_idx = tt::CBIndex::c_5;
    const uint32_t cb_squared_page_size = tile_size;
    const uint32_t cb_squared_num_pages = Wt;
    tt::tt_metal::create_cb(
        cb_squared_idx, program, core_range_set, cb_squared_page_size, cb_squared_num_pages, cb_data_format);

    // CB c_6: Variance tile
    constexpr uint32_t cb_variance_idx = tt::CBIndex::c_6;
    const uint32_t cb_variance_page_size = tile_size;
    const uint32_t cb_variance_num_pages = 1;
    tt::tt_metal::create_cb(
        cb_variance_idx, program, core_range_set, cb_variance_page_size, cb_variance_num_pages, cb_data_format);

    // CB c_7: Epsilon scalar tile
    constexpr uint32_t cb_epsilon_idx = tt::CBIndex::c_7;
    const uint32_t cb_epsilon_page_size = tile_size;
    const uint32_t cb_epsilon_num_pages = 1;
    tt::tt_metal::create_cb(
        cb_epsilon_idx, program, core_range_set, cb_epsilon_page_size, cb_epsilon_num_pages, cb_data_format);

    // CB c_8: Rsqrt result tile
    constexpr uint32_t cb_rsqrt_idx = tt::CBIndex::c_8;
    const uint32_t cb_rsqrt_page_size = tile_size;
    const uint32_t cb_rsqrt_num_pages = 1;
    tt::tt_metal::create_cb(
        cb_rsqrt_idx, program, core_range_set, cb_rsqrt_page_size, cb_rsqrt_num_pages, cb_data_format);

    // CB c_9: Standardized/output tiles (reused for Phase 8 and Phase 11 output)
    constexpr uint32_t cb_standardized_idx = tt::CBIndex::c_9;
    const uint32_t cb_standardized_page_size = tile_size;
    const uint32_t cb_standardized_num_pages = Wt;
    tt::tt_metal::create_cb(
        cb_standardized_idx,
        program,
        core_range_set,
        cb_standardized_page_size,
        cb_standardized_num_pages,
        cb_data_format);

    // CB c_10: Gamma RM sticks (read once)
    constexpr uint32_t cb_gamma_rm_idx = tt::CBIndex::c_10;
    const uint32_t cb_gamma_rm_page_size = tile_size;
    const uint32_t cb_gamma_rm_num_pages = Wt;
    tt::tt_metal::create_cb(
        cb_gamma_rm_idx, program, core_range_set, cb_gamma_rm_page_size, cb_gamma_rm_num_pages, cb_data_format);

    // CB c_11: Gamma tiled (program lifetime)
    constexpr uint32_t cb_gamma_tiled_idx = tt::CBIndex::c_11;
    const uint32_t cb_gamma_tiled_page_size = tile_size;
    const uint32_t cb_gamma_tiled_num_pages = Wt;
    tt::tt_metal::create_cb(
        cb_gamma_tiled_idx,
        program,
        core_range_set,
        cb_gamma_tiled_page_size,
        cb_gamma_tiled_num_pages,
        cb_data_format);

    // CB c_12: Beta RM sticks (read once)
    constexpr uint32_t cb_beta_rm_idx = tt::CBIndex::c_12;
    const uint32_t cb_beta_rm_page_size = tile_size;
    const uint32_t cb_beta_rm_num_pages = Wt;
    tt::tt_metal::create_cb(
        cb_beta_rm_idx, program, core_range_set, cb_beta_rm_page_size, cb_beta_rm_num_pages, cb_data_format);

    // CB c_13: Beta tiled (program lifetime)
    constexpr uint32_t cb_beta_tiled_idx = tt::CBIndex::c_13;
    const uint32_t cb_beta_tiled_page_size = tile_size;
    const uint32_t cb_beta_tiled_num_pages = Wt;
    tt::tt_metal::create_cb(
        cb_beta_tiled_idx, program, core_range_set, cb_beta_tiled_page_size, cb_beta_tiled_num_pages, cb_data_format);

    // CB c_14: Scaled output (Phase 10 gamma multiply output)
    constexpr uint32_t cb_scaled_idx = tt::CBIndex::c_14;
    const uint32_t cb_scaled_page_size = tile_size;
    const uint32_t cb_scaled_num_pages = Wt;
    tt::tt_metal::create_cb(
        cb_scaled_idx, program, core_range_set, cb_scaled_page_size, cb_scaled_num_pages, cb_data_format);

    // CB c_16: Output RM sticks (double-buffered)
    constexpr uint32_t cb_out_rm_idx = tt::CBIndex::c_16;
    const uint32_t cb_out_rm_page_size = tile_size;
    const uint32_t cb_out_rm_num_pages = buffering_factor * Wt;
    tt::tt_metal::create_cb(
        cb_out_rm_idx, program, core_range_set, cb_out_rm_page_size, cb_out_rm_num_pages, cb_data_format);

    // ============================================================
    // 5. Throw before kernel creation (Stage 5 boundary)
    // ============================================================
    // Suppress unused variable warnings for Stage 5 (used in Stage 6)
    (void)operation_attributes;
    (void)Ht;
    (void)input_stick_size_aligned;
    (void)gamma_stick_size_aligned;
    (void)beta_stick_size_aligned;
    (void)output_stick_size_aligned;

    TT_THROW("layer_norm_w_rm: Kernel creation not yet implemented (Stage 6)");
}

void LayerNormWRmProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const LayerNormWRmParams& operation_attributes,
    const LayerNormWRmInputs& tensor_args,
    Tensor& tensor_return_value) {
    (void)cached_program;
    (void)operation_attributes;
    (void)tensor_args;
    (void)tensor_return_value;
    // Stub - update runtime arguments for cached program
    // This will update tensor buffer addresses when program is reused
}

}  // namespace ttnn::operations::normalization::layer_norm_w_rm::program
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/matmul/device/config/matmul_program_config.hpp"
#include "ttnn/operations/matmul/device/utilities/matmul_utilities.hpp"
#include "ttnn/types.hpp"
#include <ranges>

namespace ttnn::operations::matmul {

namespace {

// Ensure there are always symmetrical values. Different paths use different
// index ordering (0,1 vs 1,0) to meet test PCC requirements.
constexpr std::array<std::tuple<uint32_t, uint32_t>, 20> SUBBLOCK_HW_CHOICES = {{
    {4, 2}, {2, 4}, {8, 1}, {1, 8},  // subblock_hw = 8
    {7, 1}, {1, 7},                  // subblock_hw = 7
    {3, 2}, {2, 3}, {6, 1}, {1, 6},  // subblock_hw = 6
    {5, 1}, {1, 5},                  // subblock_hw = 5
    {2, 2}, {4, 1}, {1, 4},          // subblock_hw = 4
    {3, 1}, {1, 3},                  // subblock_hw = 3
    {2, 1}, {1, 2},                  // subblock_hw = 2
    {1, 1},                          // subblock_hw = 1
}};

// This function helps determine if an optimised 1D MM config will provide perf benefits for a matmul
bool is_narrow_shape(uint32_t height, uint32_t width, bool all_dram) {
    constexpr uint32_t NARROW_SHAPE_RATIO_THRESHOLD = 8;
    uint32_t height_width_ratio = (height > width) ? height / width : width / height;

    // Check if tensor is actually narrow and will benefit from 1D config
    if (height_width_ratio > NARROW_SHAPE_RATIO_THRESHOLD) {
        return true;
    }

    // MMs that are entirely in DRAM but with a dimension smaller than the tile size will benefit from 1D config
    if (all_dram) {
        return height <= ttnn::types::TILE_SIZE || width <= ttnn::types::TILE_SIZE;
    }

    return false;
}

inline uint32_t get_per_core_factor(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const bool transpose_a,
    const bool transpose_b,
    const uint32_t bias_single_tile_size,
    uint32_t in0_block_w,
    const std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config,
    const tt::tt_metal::DataType output_dtype) {
    uint32_t max_l1_space = utilities::get_max_l1_space(input_tensor_a);
    for (uint32_t per_core_factor = 16; per_core_factor > 1; per_core_factor /= 2) {
        uint32_t size = utilities::get_estimated_size_of_cbs(
            per_core_factor,
            per_core_factor,
            in0_block_w,
            input_tensor_a,
            input_tensor_b,
            transpose_a,
            transpose_b,
            utilities::estimate_interm_tile_size(compute_kernel_config, output_dtype),
            bias_single_tile_size);
        if (size < max_l1_space) {
            return per_core_factor;
        }
    }
    return 1;
}

std::vector<uint32_t> get_multi_dim_per_core_factor(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const bool transpose_a,
    const bool transpose_b,
    const uint32_t bias_single_tile_size,
    uint32_t per_core_M,
    uint32_t per_core_N,
    uint32_t in0_block_w,
    uint32_t interm_cb_size,
    const bool adjust_in0_block_w) {
    uint32_t max_l1_space = utilities::get_max_l1_space(input_tensor_a);
    uint32_t size = utilities::get_estimated_size_of_cbs(
        per_core_M,
        per_core_N,
        in0_block_w,
        input_tensor_a,
        input_tensor_b,
        transpose_a,
        transpose_b,
        interm_cb_size,
        bias_single_tile_size);
    if (size < max_l1_space) {
        return {per_core_M, per_core_N, in0_block_w};
    }

    std::vector<uint32_t> m_factors = {per_core_M, 1};
    std::vector<uint32_t> n_factors = {per_core_N, 1};
    for (uint32_t per_core_factor_m = per_core_M / 2; per_core_factor_m > 1; per_core_factor_m--) {
        if (per_core_M % per_core_factor_m == 0) {
            m_factors.push_back(per_core_factor_m);
        }
    }
    for (uint32_t per_core_factor_n = per_core_N / 2; per_core_factor_n > 1; per_core_factor_n--) {
        if (per_core_N % per_core_factor_n == 0) {
            n_factors.push_back(per_core_factor_n);
        }
    }
    // Insert into ordered map, over write entry if new one is closer to a square
    // (smallest ratio closest to 1).
    std::map<uint32_t, std::tuple<uint32_t, uint32_t>> factors;
    for (uint32_t per_core_factor_m : m_factors) {
        for (uint32_t per_core_factor_n : n_factors) {
            uint32_t multiple = per_core_factor_m * per_core_factor_n;
            float ratio = (float)std::max(per_core_factor_m, per_core_factor_n) /
                          (float)std::min(per_core_factor_m, per_core_factor_n);
            auto entry = factors.find(multiple);
            bool add = true;
            if (entry != factors.end()) {
                auto [existing_m, existing_n] = entry->second;
                float existing_ratio =
                    (float)std::max(existing_m, existing_n) / (float)std::min(existing_m, existing_n);
                if (existing_ratio < ratio) {
                    add = false;
                }
            }
            if (add) {
                factors[multiple] = {per_core_factor_m, per_core_factor_n};
            }
        }
    }

    // Find what fits, going from largest to smallest m*n. Have k in outer loop to
    // try to maintain per_core_factor_k.
    uint32_t min_per_core_factor_k = adjust_in0_block_w ? 1 : in0_block_w;
    for (uint32_t per_core_factor_k = in0_block_w; per_core_factor_k >= min_per_core_factor_k; per_core_factor_k--) {
        if (in0_block_w % per_core_factor_k != 0) {
            continue;
        }
        for (const auto& factor : std::ranges::reverse_view(factors)) {
            uint32_t per_core_factor_m = std::get<0>(factor.second);
            uint32_t per_core_factor_n = std::get<1>(factor.second);

            size = utilities::get_estimated_size_of_cbs(
                per_core_factor_m,
                per_core_factor_n,
                per_core_factor_k,
                input_tensor_a,
                input_tensor_b,
                transpose_a,
                transpose_b,
                interm_cb_size,
                bias_single_tile_size);
            if (size < max_l1_space) {
                return {per_core_factor_m, per_core_factor_n, per_core_factor_k};
            }
        }
    }
    return {1, 1, 1};
}

std::tuple<uint32_t, uint32_t> get_subblock_sizes(
    uint32_t m_tiles_per_core, uint32_t n_tiles_per_core, bool fp32_dest_acc_en) {
    uint32_t out_subblock_h, out_subblock_w;
    for (const auto& subblock_hw : SUBBLOCK_HW_CHOICES) {
        out_subblock_w = std::get<0>(subblock_hw);
        out_subblock_h = std::get<1>(subblock_hw);
        if ((out_subblock_h * out_subblock_w) <= 4 || !fp32_dest_acc_en) {
            if (m_tiles_per_core % out_subblock_h == 0 && n_tiles_per_core % out_subblock_w == 0) {
                return {out_subblock_h, out_subblock_w};
            }
        }
    }
    TT_THROW("Unable to find subblock sizes");
}

bool can_cbs_fit_in_l1(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const bool transpose_a,
    const bool transpose_b,
    const uint32_t bias_single_tile_size,
    uint32_t per_core_M,
    uint32_t per_core_N,
    uint32_t in0_block_w,
    const std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config,
    const tt::tt_metal::DataType output_dtype) {
    uint32_t max_l1_space = utilities::get_max_l1_space(input_tensor_a);
    uint32_t size = utilities::get_estimated_size_of_cbs(
        per_core_M,
        per_core_N,
        in0_block_w,
        input_tensor_a,
        input_tensor_b,
        transpose_a,
        transpose_b,
        utilities::estimate_interm_tile_size(compute_kernel_config, output_dtype),
        bias_single_tile_size);
    return size < max_l1_space;
}

/*********************************    FACTORIES    *****************************************************/

MatmulProgramConfig create_matmul_1d_systolic_array_program_config(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const bool transpose_a,
    const bool transpose_b,
    const uint32_t bias_single_tile_size,
    const CoreCoord& core_coord,
    const std::optional<const unary::UnaryWithParam>& fused_activation,
    const bool fp32_dest_acc_en,
    const TensorMemoryLayout input_layout_a,
    const std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config,
    const tt::tt_metal::DataType output_dtype) {
    using namespace tt;
    const auto& a_padded_shape = utilities::get_matmul_tensor_padded_shape(input_tensor_a, transpose_a);
    const auto& b_padded_shape = utilities::get_matmul_tensor_padded_shape(input_tensor_b, transpose_b);
    auto k_size = a_padded_shape[-1];
    auto m_size = a_padded_shape[-2];
    auto n_size = b_padded_shape[-1];
    uint32_t batch_size_a = get_batch_size(a_padded_shape);
    uint32_t batch_size_b = get_batch_size(b_padded_shape);
    TT_FATAL(
        batch_size_b == 1,
        "Second input cannot be currently batched when running matmul using "
        "1d systolic array");
    TT_FATAL(
        (batch_size_a * m_size) % ttnn::TILE_SIZE == 0 && k_size % ttnn::TILE_SIZE == 0 &&
            n_size % ttnn::TILE_SIZE == 0,
        "The last two dimensions of the first tensor and the last dimension "
        "of the second tensor must be a multiple of "
        "tile size");
    uint32_t batch_and_m_tiles = (batch_size_a * m_size) / ttnn::TILE_SIZE;
    uint32_t k_tiles = k_size / ttnn::TILE_SIZE;
    uint32_t n_tiles = n_size / ttnn::TILE_SIZE;
    uint32_t num_cores = core_coord.x * core_coord.y;
    bool is_tall = batch_and_m_tiles > n_tiles;
    // specific 1D mcasts require specific layout types. Override accordingly.
    if (input_layout_a == TensorMemoryLayout::HEIGHT_SHARDED) {
        is_tall = true;
    } else if (input_layout_a == TensorMemoryLayout::WIDTH_SHARDED) {
        is_tall = false;
    }

    bool is_wide = !is_tall;
    uint32_t batch_and_m_tiles_per_core;
    uint32_t k_tiles_per_core;
    uint32_t n_tiles_per_core;
    if (is_tall) {
        batch_and_m_tiles_per_core = div_up(batch_and_m_tiles, num_cores);
        k_tiles_per_core = div_up(k_tiles, num_cores);
        n_tiles_per_core = n_tiles;
    } else {
        batch_and_m_tiles_per_core = batch_and_m_tiles;
        k_tiles_per_core = div_up(k_tiles, num_cores);
        n_tiles_per_core = div_up(n_tiles, num_cores);
    }
    while (k_tiles % k_tiles_per_core != 0) {
        k_tiles_per_core -= 1;
    }
    auto mutlti_dim_per_core_factor = get_multi_dim_per_core_factor(
        input_tensor_a,
        input_tensor_b,
        transpose_a,
        transpose_b,
        bias_single_tile_size,
        batch_and_m_tiles_per_core,
        n_tiles_per_core,
        k_tiles_per_core,
        utilities::estimate_interm_tile_size(compute_kernel_config, output_dtype),
        /*adjust_in0_block_w=*/false);
    uint32_t out_block_h = mutlti_dim_per_core_factor[0];
    uint32_t out_block_w = mutlti_dim_per_core_factor[1];

    auto matmul_params = get_subblock_sizes(out_block_h, out_block_w, fp32_dest_acc_en);
    uint32_t out_subblock_h = std::get<0>(matmul_params);
    uint32_t out_subblock_w = std::get<1>(matmul_params);
    return MatmulMultiCoreReuseMultiCast1DProgramConfig{
        .compute_with_storage_grid_size = {core_coord.x, core_coord.y},
        .in0_block_w = k_tiles_per_core,
        .out_subblock_h = out_subblock_h,
        .out_subblock_w = out_subblock_w,
        .out_block_h = out_block_h,
        .out_block_w = out_block_w,
        .per_core_M = batch_and_m_tiles_per_core,
        .per_core_N = n_tiles_per_core,
        .fuse_batch = true,
        .fused_activation = fused_activation,
        .mcast_in0 = is_wide,
    };
}

MatmulMultiCoreReuseMultiCast1DProgramConfig get_mcast_1d_config(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const bool transpose_a,
    const bool transpose_b,
    const uint32_t bias_single_tile_size,
    const bool fuse_batch,
    const std::optional<unary::UnaryWithParam>& fused_activation,
    const bool mcast_in0,
    const bool out_sharded,
    const std::optional<const CoreCoord> compute_with_storage_grid_size,
    const std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config,
    const tt::tt_metal::DataType output_dtype,
    const bool /*all_dram_interleaved*/) {
    using namespace tt;
    auto* device = input_tensor_a.device();
    auto grid_size = compute_with_storage_grid_size.value_or(device->compute_with_storage_grid_size());

    const auto& a_shape_padded = utilities::get_matmul_tensor_padded_shape(input_tensor_a, transpose_a);
    const auto& b_shape_padded = utilities::get_matmul_tensor_padded_shape(input_tensor_b, transpose_b);
    auto in0_tile = utilities::get_matmul_tile(input_tensor_a, transpose_a);
    auto in1_tile = utilities::get_matmul_tile(input_tensor_b, transpose_b);

    const auto M = utilities::get_M_dim(a_shape_padded, /*tile=*/std::nullopt, fuse_batch);
    const auto K = utilities::get_K_dim(a_shape_padded, /*tile=*/std::nullopt);
    const auto N = utilities::get_N_dim(b_shape_padded, /*tile=*/std::nullopt);
    uint32_t per_core_M, per_core_N;
    if (mcast_in0) {
        per_core_M = M / in0_tile.get_height();
        per_core_N = div_up(div_up(N, grid_size.x * grid_size.y), in1_tile.get_width());
    } else {
        per_core_M = div_up(div_up(M, grid_size.x * grid_size.y), in0_tile.get_height());
        per_core_N = N / in1_tile.get_width();
    }
    uint32_t in0_block_w = K / in0_tile.get_width() % 2 == 0 ? 2 : 1;
    bool per_core_N_equals_subblock_w_constraint = out_sharded && !mcast_in0;
    bool per_core_M_equals_subblock_h_constraint = out_sharded && mcast_in0;
    bool fp32_dest_acc_en = get_fp32_dest_acc_en(compute_kernel_config);

    auto mutlti_dim_per_core_factor = get_multi_dim_per_core_factor(
        input_tensor_a,
        input_tensor_b,
        transpose_a,
        transpose_b,
        bias_single_tile_size,
        per_core_M,
        per_core_N,
        in0_block_w,
        utilities::estimate_interm_tile_size(compute_kernel_config, output_dtype),
        /*adjust_in0_block_w=*/false);
    uint32_t out_block_h = mutlti_dim_per_core_factor[0];
    uint32_t out_block_w = mutlti_dim_per_core_factor[1];

    auto subblock_hw = bmm_op_utils::get_matmul_subblock_params(
        out_block_h,
        out_block_w,
        per_core_M_equals_subblock_h_constraint,
        per_core_N_equals_subblock_w_constraint,
        fp32_dest_acc_en);
    auto out_subblock_h = std::get<0>(subblock_hw);
    auto out_subblock_w = std::get<1>(subblock_hw);

    return MatmulMultiCoreReuseMultiCast1DProgramConfig{
        .compute_with_storage_grid_size = grid_size,
        .in0_block_w = in0_block_w,
        .out_subblock_h = out_subblock_h,
        .out_subblock_w = out_subblock_w,
        .out_block_h = out_block_h,
        .out_block_w = out_block_w,
        .per_core_M = per_core_M,
        .per_core_N = per_core_N,
        .fuse_batch = fuse_batch,
        .fused_activation = fused_activation,
        .mcast_in0 = mcast_in0};
}

MatmulProgramConfig create_matmul_program_config(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const bool transpose_a,
    const bool transpose_b,
    const uint32_t bias_single_tile_size,
    const std::optional<const CoreCoord> user_core_coord,
    const std::optional<unary::UnaryWithParam>& fused_activation,
    const std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config,
    const MemoryConfig& mem_config,
    const tt::tt_metal::DataType output_dtype) {
    using namespace tt;
    const auto a_shape = utilities::get_matmul_tensor_logical_shape(input_tensor_a, transpose_a);
    const auto b_shape = utilities::get_matmul_tensor_logical_shape(input_tensor_b, transpose_b);
    const auto& a_padded_shape = utilities::get_matmul_tensor_padded_shape(input_tensor_a, transpose_a);
    const auto& b_padded_shape = utilities::get_matmul_tensor_padded_shape(input_tensor_b, transpose_b);
    auto a_layout = input_tensor_a.memory_config().memory_layout();
    auto inteneded_k_size_of_a = a_shape[-1];
    auto inteneded_k_size_of_b = b_shape[-2];
    auto k_size = a_padded_shape[-1];
    auto m_size = a_padded_shape[-2];
    auto n_size = b_padded_shape[-1];
    uint32_t batch_size_a = get_batch_size(a_padded_shape);
    uint32_t batch_size_b = get_batch_size(b_padded_shape);
    bool input_b_is_batched = batch_size_b > 1;
    bool any_size_within_tile = k_size <= ttnn::TILE_SIZE || m_size <= ttnn::TILE_SIZE || n_size <= ttnn::TILE_SIZE;
    const auto& input_tensor_a_memory_config = input_tensor_a.memory_config();
    const auto& input_tensor_b_memory_config = input_tensor_b.memory_config();
    bool fp32_dest_acc_en = get_fp32_dest_acc_en(compute_kernel_config);
    bool a_is_sharded = input_tensor_a.is_sharded();
    TT_FATAL(inteneded_k_size_of_a == inteneded_k_size_of_b, "The k dimension does not match between tensors");
    TT_FATAL(
        (batch_size_a * m_size) % ttnn::TILE_SIZE == 0 && k_size % ttnn::TILE_SIZE == 0 &&
            n_size % ttnn::TILE_SIZE == 0,
        "The last two dimensions of the first tensor and the last dimension "
        "of the second tensor must be a multiple of "
        "tile size");
    auto core_coord = input_tensor_a.device()->compute_with_storage_grid_size();
    bool has_user_core_coord = user_core_coord.has_value();
    if (has_user_core_coord) {
        auto x = user_core_coord.value().x;
        auto y = user_core_coord.value().y;
        if (x <= core_coord.x && y <= core_coord.y) {
            core_coord = user_core_coord.value();
        }
    }

    uint32_t m_tiles_per_core;
    uint32_t n_tiles_per_core;
    uint32_t k_tiles_per_core;
    if (input_b_is_batched) {
        TT_FATAL(!fused_activation.has_value(), "Cannot use activation with batched input b");
        if (!a_is_sharded && !input_tensor_b.is_sharded()) {
            m_tiles_per_core = div_up(m_size, ttnn::TILE_SIZE);
            n_tiles_per_core = div_up(n_size, ttnn::TILE_SIZE);
            k_tiles_per_core = 1;  // TODO(arakhmati): Can it be more than 1 without
                                   // running out of memory?
            if (!can_cbs_fit_in_l1(
                    input_tensor_a,
                    input_tensor_b,
                    transpose_a,
                    transpose_b,
                    bias_single_tile_size,
                    m_tiles_per_core,
                    n_tiles_per_core,
                    k_tiles_per_core,
                    compute_kernel_config,
                    output_dtype)) {
                return create_simple_matmul_program_config(
                    input_tensor_a,
                    input_tensor_b,
                    transpose_a,
                    transpose_b,
                    bias_single_tile_size,
                    compute_kernel_config,
                    core_coord,
                    mem_config,
                    output_dtype);
            }
        } else if (a_is_sharded) {
            TT_FATAL(
                a_layout != TensorMemoryLayout::WIDTH_SHARDED,
                "MatmulMultiCoreReuseProgramConfig: Input A cannot be width sharded, got layout: {}",
                a_layout);
            auto shard_shape = input_tensor_a_memory_config.shard_spec().value().shape;
            uint32_t n = b_shape[-1] / ttnn::TILE_SIZE;
            m_tiles_per_core = shard_shape[0] / ttnn::TILE_SIZE;
            n_tiles_per_core = n;
            k_tiles_per_core = shard_shape[1] / ttnn::TILE_SIZE;
        } else {
            TT_FATAL(
                input_tensor_b_memory_config.memory_layout() != TensorMemoryLayout::WIDTH_SHARDED,
                "MatmulMultiCoreReuseProgramConfig: Input B cannot be width sharded, got layout: {}",
                input_tensor_b_memory_config.memory_layout());
            auto shard_shape = input_tensor_b_memory_config.shard_spec().value().shape;
            m_tiles_per_core = div_up(m_size, ttnn::TILE_SIZE);
            n_tiles_per_core = shard_shape[1] / ttnn::TILE_SIZE;
            k_tiles_per_core = 1;
        }

        auto matmul_params = get_subblock_sizes(m_tiles_per_core, n_tiles_per_core, fp32_dest_acc_en);
        uint32_t out_subblock_h = std::get<0>(matmul_params);
        uint32_t out_subblock_w = std::get<1>(matmul_params);

        return MatmulMultiCoreReuseProgramConfig{
            .compute_with_storage_grid_size = {core_coord.x, core_coord.y},
            .in0_block_w = k_tiles_per_core,
            .out_subblock_h = out_subblock_h,
            .out_subblock_w = out_subblock_w,
            .per_core_M = m_tiles_per_core,
            .per_core_N = n_tiles_per_core,
        };
    }

    auto height = batch_size_a * m_size;
    auto width = n_size;
    bool a_is_block_sharded = a_layout == TensorMemoryLayout::BLOCK_SHARDED;
    if (is_narrow_shape(height, width, false) || any_size_within_tile) {
        if (!a_is_block_sharded) {
            return create_matmul_1d_systolic_array_program_config(
                input_tensor_a,
                input_tensor_b,
                transpose_a,
                transpose_b,
                bias_single_tile_size,
                core_coord,
                fused_activation,
                fp32_dest_acc_en,
                a_layout,
                compute_kernel_config,
                output_dtype);
        }
    }
    if (!a_is_sharded) {
        m_tiles_per_core = (uint32_t)std::ceil((((double)batch_size_a * m_size) / ttnn::TILE_SIZE) / core_coord.y);
        n_tiles_per_core = (uint32_t)std::ceil((double)n_size / ttnn::TILE_SIZE / core_coord.x);
        k_tiles_per_core = 4;  // TODO(arakhmati): What is a good starting point?
        while ((k_size / ttnn::TILE_SIZE) % k_tiles_per_core != 0) {
            k_tiles_per_core -= 1;
        }
    } else {
        if (!a_is_block_sharded) {
            return create_matmul_1d_systolic_array_program_config(
                input_tensor_a,
                input_tensor_b,
                transpose_a,
                transpose_b,
                bias_single_tile_size,
                core_coord,
                fused_activation,
                fp32_dest_acc_en,
                a_layout,
                compute_kernel_config,
                output_dtype);
        }
        uint32_t k = a_padded_shape[-1] / ttnn::TILE_SIZE;
        uint32_t n = b_padded_shape[-1] / ttnn::TILE_SIZE;
        auto shard_shape = input_tensor_a_memory_config.shard_spec().value().shape;
        m_tiles_per_core = shard_shape[0] / ttnn::TILE_SIZE;
        n_tiles_per_core = (n * shard_shape[1]) / (k * ttnn::TILE_SIZE);
        k_tiles_per_core = std::gcd(shard_shape[1] / ttnn::TILE_SIZE, k);
    }

    n_tiles_per_core = std::max(n_tiles_per_core, (unsigned int)1);

    auto mutlti_dim_per_core_factor = get_multi_dim_per_core_factor(
        input_tensor_a,
        input_tensor_b,
        transpose_a,
        transpose_b,
        bias_single_tile_size,
        m_tiles_per_core,
        n_tiles_per_core,
        k_tiles_per_core,
        utilities::estimate_interm_tile_size(compute_kernel_config, output_dtype),
        /*adjust_in0_block_w=*/false);
    uint32_t out_block_h = mutlti_dim_per_core_factor[0];
    uint32_t out_block_w = mutlti_dim_per_core_factor[1];

    auto matmul_params = get_subblock_sizes(out_block_h, out_block_w, fp32_dest_acc_en);
    uint32_t out_subblock_h = std::get<0>(matmul_params);
    uint32_t out_subblock_w = std::get<1>(matmul_params);
    bool transpose_mcast =
        a_is_block_sharded && input_tensor_a.shard_spec().value().orientation == ShardOrientation::COL_MAJOR;
    if (out_subblock_w != n_tiles_per_core) {
        out_subblock_h = 1;
    }

    return MatmulMultiCoreReuseMultiCastProgramConfig{
        .compute_with_storage_grid_size = {core_coord.x, core_coord.y},
        .in0_block_w = k_tiles_per_core,
        .out_subblock_h = out_subblock_h,
        .out_subblock_w = out_subblock_w,
        .out_block_h = out_block_h,
        .out_block_w = out_block_w,
        .per_core_M = m_tiles_per_core,
        .per_core_N = n_tiles_per_core,
        .transpose_mcast = transpose_mcast,
        .fused_activation = fused_activation,
    };
}

// TODO: Only supports sharded matmul for now; infer most matmul params from
// shard spec
MatmulProgramConfig get_matmul_program_config(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const bool transpose_a,
    const bool transpose_b,
    const uint32_t bias_single_tile_size,
    const MemoryConfig& output_mem_config,
    const std::optional<unary::UnaryWithParam>& fused_activation,
    const bool matmul,
    const std::optional<const CoreCoord> user_core_coord,
    const std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config,
    const tt::tt_metal::DataType output_dtype) {
    using namespace tt;
    TT_FATAL(input_tensor_a.is_sharded(), "Input tensor A must be sharded");
    bool fp32_dest_acc_en = get_fp32_dest_acc_en(compute_kernel_config);
    // TODO: allow overwriting of grid size by user_core_coord after allowing
    // support of arbitrary compute grid and more generic sharded output tensor
    // creation
    auto grid_size = input_tensor_a.shard_spec().value().grid.bounding_box().grid_size();

    const auto& a_shape_padded = utilities::get_matmul_tensor_padded_shape(input_tensor_a, transpose_a);
    const auto& b_shape_padded = utilities::get_matmul_tensor_padded_shape(input_tensor_b, transpose_b);
    auto in0_tile = utilities::get_matmul_tile(input_tensor_a, transpose_a);
    auto in1_tile = utilities::get_matmul_tile(input_tensor_b, transpose_b);

    // MCAST matmuls only support input_b in INTERLEAVED
    if (matmul) {
        TT_FATAL(
            input_tensor_b.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
            "Input tensor B must have INTERLEAVED memory layout, got: {}",
            input_tensor_b.memory_config().memory_layout());
        if ((input_tensor_a.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED or
             input_tensor_a.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED) and
            (grid_size.x > 1 or grid_size.y > 1)) {
            TT_FATAL(
                input_tensor_a.shard_spec().value().orientation == ShardOrientation::ROW_MAJOR,
                "Input tensor A must have ROW_MAJOR shard orientation, got: {}",
                input_tensor_a.shard_spec().value().orientation);

            bool per_core_N_equals_subblock_w_constraint = false;
            if (output_mem_config.is_sharded()) {
                TT_FATAL(
                    input_tensor_a.memory_config().buffer_type() == output_mem_config.buffer_type(),
                    "Input A and output buffer types must match, got input: {} vs output: {}",
                    input_tensor_a.memory_config().buffer_type(),
                    output_mem_config.buffer_type());
                TT_FATAL(
                    input_tensor_a.memory_config().memory_layout() == output_mem_config.memory_layout(),
                    "Input A and output memory layouts must match, got input: {} vs output: {}",
                    input_tensor_a.memory_config().memory_layout(),
                    output_mem_config.memory_layout());
                per_core_N_equals_subblock_w_constraint = true;
            }

            const auto M = utilities::get_M_dim(a_shape_padded, in0_tile, /*fuse_batch=*/true);
            const auto K = utilities::get_K_dim(a_shape_padded, in0_tile);
            const auto N = utilities::get_N_dim(b_shape_padded, in1_tile);
            auto shard_shape = input_tensor_a.shard_spec().value().shape;

            bool mcast_in0;
            uint32_t per_core_M;
            uint32_t per_core_N;
            uint32_t in0_block_w;
            if (input_tensor_a.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED) {
                mcast_in0 = true;
                per_core_M = M;
                per_core_N = div_up(N, input_tensor_a.shard_spec().value().grid.num_cores());
                in0_block_w = std::gcd(shard_shape[1] / in0_tile.get_width(), K);
            } else if (input_tensor_a.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED) {
                mcast_in0 = false;
                per_core_M = shard_shape[0] / in0_tile.get_height();
                per_core_N = N;  // Only necessary if output is sharded; otherwise, can
                                 // set this to be < N
                in0_block_w = K;
            } else {
                TT_THROW(
                    "Input tensor must be WIDTH or HEIGHT sharded for 1D mcast "
                    "matmul!");
            }

            auto mutlti_dim_per_core_factor = get_multi_dim_per_core_factor(
                input_tensor_a,
                input_tensor_b,
                transpose_a,
                transpose_b,
                bias_single_tile_size,
                per_core_M,
                per_core_N,
                in0_block_w,
                utilities::estimate_interm_tile_size(compute_kernel_config, output_dtype),
                /*adjust_in0_block_w=*/false);
            uint32_t out_block_h = mutlti_dim_per_core_factor[0];
            uint32_t out_block_w = mutlti_dim_per_core_factor[1];

            auto subblock_hw = bmm_op_utils::get_matmul_subblock_params(
                out_block_h, out_block_w, false, per_core_N_equals_subblock_w_constraint, fp32_dest_acc_en);
            auto out_subblock_h = std::get<0>(subblock_hw);
            auto out_subblock_w = std::get<1>(subblock_hw);

            return MatmulMultiCoreReuseMultiCast1DProgramConfig{
                .compute_with_storage_grid_size = grid_size,
                .in0_block_w = in0_block_w,
                .out_subblock_h = out_subblock_h,
                .out_subblock_w = out_subblock_w,
                .out_block_h = out_block_h,
                .out_block_w = out_block_w,
                .per_core_M = per_core_M,
                .per_core_N = per_core_N,
                .fuse_batch = true,
                .fused_activation = fused_activation,
                .mcast_in0 = mcast_in0,
            };
        }
        if (input_tensor_a.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED and
            (grid_size.x > 1 and grid_size.y > 1)) {
            bool transpose_mcast = input_tensor_a.shard_spec().value().orientation == ShardOrientation::COL_MAJOR;

            bool per_core_N_equals_subblock_w_constraint = false;
            if (output_mem_config.is_sharded()) {
                TT_FATAL(
                    input_tensor_a.memory_config().buffer_type() == output_mem_config.buffer_type(),
                    "Input A and output buffer types must match, got input: {} vs output: {}",
                    input_tensor_a.memory_config().buffer_type(),
                    output_mem_config.buffer_type());
                TT_FATAL(
                    input_tensor_a.memory_config().memory_layout() == output_mem_config.memory_layout(),
                    "Input A and output memory layouts must match, got input: {} vs output: {}",
                    input_tensor_a.memory_config().memory_layout(),
                    output_mem_config.memory_layout());
                per_core_N_equals_subblock_w_constraint = true;
            }

            const auto M = utilities::get_M_dim(a_shape_padded, in0_tile, /*fuse_batch=*/true);
            const auto K = utilities::get_K_dim(a_shape_padded, in0_tile);
            const auto N = utilities::get_N_dim(b_shape_padded, in1_tile);

            auto shard_shape = input_tensor_a.shard_spec().value().shape;
            uint32_t virtual_x = transpose_mcast ? grid_size.y : grid_size.x;
            uint32_t virtual_y = transpose_mcast ? grid_size.x : grid_size.y;
            bool cores_along_x_match_grid_size = virtual_x == (K / (shard_shape[1] / in0_tile.get_width()));
            bool cores_along_y_match_grid_size = virtual_y == (M / (shard_shape[0] / in0_tile.get_height()));
            TT_FATAL(
                cores_along_y_match_grid_size || virtual_y == div_up(M, (shard_shape[0] / in0_tile.get_height())),
                "Num cores along y ({}) must match provided grid size ({}) or divided up size ({})",
                virtual_y,
                M / (shard_shape[0] / in0_tile.get_height()),
                div_up(M, (shard_shape[0] / in0_tile.get_height())));
            TT_FATAL(
                cores_along_x_match_grid_size || virtual_x == div_up(K, (shard_shape[1] / in0_tile.get_width())),
                "Num cores along x ({}) must match provided grid size ({}) or divided up size ({})",
                virtual_x,
                K / (shard_shape[1] / in0_tile.get_width()),
                div_up(K, (shard_shape[1] / in0_tile.get_width())));

            uint32_t per_core_M = div_up(M, virtual_y);
            uint32_t per_core_N = div_up(N, virtual_x);
            uint32_t in0_block_w =
                cores_along_x_match_grid_size ? std::gcd(shard_shape[1] / in0_tile.get_width(), K) : 1;

            auto mutlti_dim_per_core_factor = get_multi_dim_per_core_factor(
                input_tensor_a,
                input_tensor_b,
                transpose_a,
                transpose_b,
                bias_single_tile_size,
                per_core_M,
                per_core_N,
                in0_block_w,
                utilities::estimate_interm_tile_size(compute_kernel_config, output_dtype),
                /*adjust_in0_block_w=*/false);
            uint32_t out_block_h = mutlti_dim_per_core_factor[0];
            uint32_t out_block_w = mutlti_dim_per_core_factor[1];

            auto subblock_hw = bmm_op_utils::get_matmul_subblock_params(
                out_block_h, out_block_w, false, per_core_N_equals_subblock_w_constraint, fp32_dest_acc_en);
            auto out_subblock_h = std::get<0>(subblock_hw);
            auto out_subblock_w = std::get<1>(subblock_hw);

            return MatmulMultiCoreReuseMultiCastProgramConfig{
                .compute_with_storage_grid_size = grid_size,
                .in0_block_w = in0_block_w,
                .out_subblock_h = out_subblock_h,
                .out_subblock_w = out_subblock_w,
                .out_block_h = out_block_h,
                .out_block_w = out_block_w,
                .per_core_M = per_core_M,
                .per_core_N = per_core_N,
                .transpose_mcast = transpose_mcast,
                .fused_activation = fused_activation,
            };
        }
    } else {
        // TODO: Need a better criteria for BMMs and
        // MatmulMultiCoreReuseProgramConfig
        TT_FATAL(
            input_tensor_a.memory_config().memory_layout() != TensorMemoryLayout::WIDTH_SHARDED,
            "Input A memory layout must not be WIDTH_SHARDED, got: {}",
            input_tensor_a.memory_config().memory_layout());

        bool per_core_N_equals_subblock_w_constraint = false;
        if (output_mem_config.is_sharded()) {
            TT_FATAL(
                input_tensor_a.memory_config().buffer_type() == output_mem_config.buffer_type(),
                "Input A and output buffer types must match, got input: {} vs output: {}",
                input_tensor_a.memory_config().buffer_type(),
                output_mem_config.buffer_type());
            TT_FATAL(
                input_tensor_a.memory_config().memory_layout() == output_mem_config.memory_layout(),
                "Input A and output memory layouts must match, got input: {} vs output: {}",
                input_tensor_a.memory_config().memory_layout(),
                output_mem_config.memory_layout());
            per_core_N_equals_subblock_w_constraint = true;
        }

        const auto N = utilities::get_N_dim(b_shape_padded, in1_tile);

        auto in0_shard_shape = input_tensor_a.shard_spec().value().shape;
        uint32_t per_core_M = in0_shard_shape[0] / in0_tile.get_height();
        uint32_t per_core_N = N;
        uint32_t in0_block_w = in0_shard_shape[1] / in0_tile.get_width();

        auto subblock_hw = bmm_op_utils::get_matmul_subblock_params(
            per_core_M, per_core_N, false, per_core_N_equals_subblock_w_constraint, fp32_dest_acc_en);
        auto out_subblock_h = std::get<0>(subblock_hw);
        auto out_subblock_w = std::get<1>(subblock_hw);

        // TODO: Temporarily allow for single core; should support bcast_batch in general
        const auto batch_size_a =
            get_batch_size(utilities::get_matmul_tensor_padded_shape(input_tensor_a, transpose_a));
        const auto batch_size_b =
            get_batch_size(utilities::get_matmul_tensor_padded_shape(input_tensor_b, transpose_b));
        bool broadcast_batch = batch_size_a > 1 and batch_size_b == 1;
        TT_FATAL(!broadcast_batch, "Batch broadcasting is not supported for the chosen program config");

        if (input_tensor_b.is_sharded()) {
            TT_FATAL(
                input_tensor_a.memory_config().buffer_type() == input_tensor_b.memory_config().buffer_type(),
                "Input A and B buffer types must match, got A: {} vs B: {}",
                input_tensor_a.memory_config().buffer_type(),
                input_tensor_b.memory_config().buffer_type());
            TT_FATAL(
                input_tensor_a.memory_config().memory_layout() == input_tensor_b.memory_config().memory_layout(),
                "Input A and B memory layouts must match, got A: {} vs B: {}",
                input_tensor_a.memory_config().memory_layout(),
                input_tensor_b.memory_config().memory_layout());
            TT_FATAL(
                input_tensor_a.shard_spec().value().grid == input_tensor_b.shard_spec().value().grid,
                "Input A and B shard grids must match");
            TT_FATAL(
                input_tensor_a.shard_spec().value().orientation == input_tensor_b.shard_spec().value().orientation,
                "Orientation mismatch a {} vs b {}.",
                input_tensor_a.shard_spec().value().orientation,
                input_tensor_b.shard_spec().value().orientation);
        }

        return MatmulMultiCoreReuseProgramConfig{
            .compute_with_storage_grid_size = grid_size,
            .in0_block_w = in0_block_w,
            .out_subblock_h = out_subblock_h,
            .out_subblock_w = out_subblock_w,
            .per_core_M = per_core_M,
            .per_core_N = per_core_N,
        };
    }
    return create_matmul_program_config(
        input_tensor_a,
        input_tensor_b,
        transpose_a,
        transpose_b,
        bias_single_tile_size,
        user_core_coord,
        fused_activation,
        compute_kernel_config,
        output_mem_config,
        output_dtype);
}

inline MatmulProgramConfig generate_matmul_program_config(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const bool transpose_a,
    const bool transpose_b,
    const uint32_t bias_single_tile_size,
    const MemoryConfig& mem_config,
    const std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<const CoreCoord> user_core_coord,
    const std::optional<unary::UnaryWithParam>& user_fused_activation,
    const bool user_run_batched,
    const tt::tt_metal::DataType output_dtype) {
    const bool has_user_grid = user_core_coord.has_value();
    if (has_user_grid || !input_tensor_a.is_sharded()) {
        CoreCoord core_coord;
        if (has_user_grid) {
            core_coord = user_core_coord.value();
            return create_matmul_program_config(
                input_tensor_a,
                input_tensor_b,
                transpose_a,
                transpose_b,
                bias_single_tile_size,
                user_core_coord,
                user_fused_activation,
                compute_kernel_config,
                mem_config,
                output_dtype);
        }
        tt::tt_metal::IDevice* device = input_tensor_a.device();
        auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
        return create_simple_matmul_program_config(
            input_tensor_a,
            input_tensor_b,
            transpose_a,
            transpose_b,
            bias_single_tile_size,
            compute_kernel_config,
            compute_with_storage_grid_size,
            mem_config,
            output_dtype);
    }
    bool bmm = user_run_batched;
    return get_matmul_program_config(
        input_tensor_a,
        input_tensor_b,
        transpose_a,
        transpose_b,
        bias_single_tile_size,
        mem_config,
        std::nullopt,
        !bmm,
        user_core_coord,
        compute_kernel_config,
        output_dtype);
}

/***************************************************************************************/

}  // namespace

namespace bmm_op_utils {
std::tuple<uint32_t, uint32_t> get_matmul_subblock_params(
    const uint32_t per_core_M,
    const uint32_t per_core_N,
    const bool per_core_M_equals_subblock_h_constraint,
    const bool per_core_N_equals_subblock_w_constraint,
    const bool fp32_dest_acc_en) {
    TT_FATAL(
        !(per_core_M_equals_subblock_h_constraint and per_core_N_equals_subblock_w_constraint),
        "Only one constraint may be true for h or w!");

    uint32_t out_subblock_h, out_subblock_w;
    for (const auto& subblock_hw : SUBBLOCK_HW_CHOICES) {
        out_subblock_h = std::get<0>(subblock_hw);
        out_subblock_w = std::get<1>(subblock_hw);
        if (fp32_dest_acc_en) {
            if ((out_subblock_h * out_subblock_w) > 4) {
                continue;  // Total number of tiles in a subblock must be less than 4
                           // when in fp32_dest_acc mode
            }
        }
        if (per_core_N_equals_subblock_w_constraint) {
            if (out_subblock_w != per_core_N || out_subblock_h != 1) {
                continue;
            }
        }
        if (per_core_M_equals_subblock_h_constraint) {
            if (out_subblock_h != per_core_M || out_subblock_w != 1) {
                continue;
            }
        }
        if (per_core_M % out_subblock_h == 0 and per_core_N % out_subblock_w == 0) {
            return {out_subblock_h, out_subblock_w};
        }
    }
    // Return basic value that should work in most cases.
    return {1, 1};
}
}  // namespace bmm_op_utils

MatmulProgramConfig get_program_config(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const bool transpose_a,
    const bool transpose_b,
    const uint32_t bias_single_tile_size,
    const ttnn::prim::MatmulParams& attributes) {
    if (attributes.program_config.has_value()) {
        return attributes.program_config.value();
    }
    auto config = generate_matmul_program_config(
        input_tensor_a,
        input_tensor_b,
        transpose_a,
        transpose_b,
        bias_single_tile_size,
        attributes.output_mem_config,
        attributes.compute_kernel_config,
        attributes.user_core_coord,
        attributes.user_fused_activation,
        attributes.user_run_batched,
        attributes.output_dtype.value_or(input_tensor_a.dtype()));
    log_debug(tt::LogOp, "Auto generated program config: {}", config);

    // Sanity checks for matmul program configs
    std::visit(
        [input_tensor_a](const auto& program_config) {
            using ProgramConfigType = std::decay_t<decltype(program_config)>;
            if constexpr (
                not std::is_same_v<ProgramConfigType, MatmulMultiCoreProgramConfig> and
                not std::is_same_v<ProgramConfigType, MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig>) {
                TT_FATAL(
                    program_config.compute_with_storage_grid_size.x <=
                        input_tensor_a.device()->compute_with_storage_grid_size().x,
                    "Number of columns in matmul compute grid exceeds maximum device "
                    "compute grid size!");
                TT_FATAL(
                    program_config.compute_with_storage_grid_size.y <=
                        input_tensor_a.device()->compute_with_storage_grid_size().y,
                    "Number of rows in matmul compute grid exceeds maximum device "
                    "compute grid size!");
                TT_FATAL(
                    program_config.compute_with_storage_grid_size.x > 0,
                    "Number of columns in matmul compute grid must be greater "
                    "than 0!");
                TT_FATAL(
                    program_config.compute_with_storage_grid_size.y > 0,
                    "Number of rows in matmul compute grid must be greater than 0!");
                TT_FATAL(program_config.in0_block_w > 0, "in0_block_w must be greater than 0!");
                TT_FATAL(program_config.out_subblock_h > 0, "out_subblock_h must be greater than 0!");
                TT_FATAL(program_config.out_subblock_w > 0, "out_subblock_w must be greater than 0!");
                TT_FATAL(program_config.per_core_M > 0, "per_core_M must be greater than 0!");
                TT_FATAL(program_config.per_core_N > 0, "per_core_N must be greater than 0!");
                TT_FATAL(
                    program_config.per_core_M % program_config.out_subblock_h == 0,
                    "per_core_M must be divisible by out_subblock_h!");
                TT_FATAL(
                    program_config.per_core_N % program_config.out_subblock_w == 0,
                    "per_core_N must be divisible by out_subblock_w!");
            }
        },
        config);
    return config;
}

MatmulProgramConfig create_simple_matmul_program_config(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const bool transpose_a,
    const bool transpose_b,
    const uint32_t bias_single_tile_size,
    const std::optional<const ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    const CoreCoord& compute_with_storage_grid_size,
    const tt::tt_metal::MemoryConfig& mem_config,
    const tt::tt_metal::DataType output_dtype) {
    using namespace tt::tt_metal;
    const auto& a_shape_padded = utilities::get_matmul_tensor_padded_shape(input_tensor_a, transpose_a);
    const auto& b_shape_padded = utilities::get_matmul_tensor_padded_shape(input_tensor_b, transpose_b);

    auto in0_tile = utilities::get_matmul_tile(input_tensor_a, transpose_a);
    auto in1_tile = utilities::get_matmul_tile(input_tensor_b, transpose_b);

    // Parameters for large matmul with reuse
    const auto Mt = utilities::get_M_dim(a_shape_padded, in0_tile, /*fuse_batch=*/false);
    const auto Kt = utilities::get_K_dim(a_shape_padded, in0_tile);
    const auto Nt = utilities::get_N_dim(b_shape_padded, in1_tile);
    uint32_t in0_block_w = 2;

    TT_FATAL(input_tensor_a.storage_type() == StorageType::DEVICE, "input tensor needs to be on device");
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    uint32_t per_core_M, per_core_N, out_subblock_h, out_subblock_w;
    uint32_t num_blocks_x, num_blocks_y;

    const bool all_interleaved = input_tensor_a.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED &&
                                 mem_config.memory_layout() == TensorMemoryLayout::INTERLEAVED &&
                                 input_tensor_b.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED;

    const bool all_dram = input_tensor_a.memory_config().buffer_type() == BufferType::DRAM &&
                          input_tensor_b.memory_config().buffer_type() == BufferType::DRAM &&
                          mem_config.buffer_type() == BufferType::DRAM;

    const bool all_dram_interleaved = all_dram && all_interleaved;

    uint32_t height = a_shape_padded[-2];
    uint32_t width = b_shape_padded[-1];
    const bool is_narrow = is_narrow_shape(height, width, all_dram);
    bool is_wide = false;
    bool is_tall = false;
    if (all_interleaved && is_narrow) {
        is_wide = width > height;
        is_tall = !is_wide;
    }

    // out_subblock h/w doesn't matter
    per_core_M = get_per_core_factor(
        input_tensor_a,
        input_tensor_b,
        transpose_a,
        transpose_b,
        bias_single_tile_size,
        in0_block_w,
        compute_kernel_config,
        output_dtype);
    per_core_N = per_core_M;

    // Calculate number of blocks along x and y; tensor dims are padded up to 512
    num_blocks_y = (Mt - 1) / per_core_M + 1;
    num_blocks_x = (Nt - 1) / per_core_N + 1;

    // MatmulMultiCoreProgramConfig does not support sharded output.
    // Reduce in0_block_w if necessary or might benefit from mcast due to size to
    // choose other configs.
    if ((mem_config.is_sharded() or num_blocks_y > 1 or num_blocks_x > 1) and Kt % in0_block_w != 0) {
        in0_block_w = 1;
    }

    auto get_core_range = [](uint32_t num_blocks_rows,
                             uint32_t num_blocks_cols,
                             uint32_t max_num_rows,
                             uint32_t max_num_cols) -> CoreCoord {
        CoreCoord core_range(0, 0);
        if (!(num_blocks_rows == 1 && num_blocks_cols == 1) && num_blocks_rows <= max_num_rows &&
            num_blocks_cols <= max_num_cols) {
            core_range.x = num_blocks_cols;
            core_range.y = num_blocks_rows;
        }
        return core_range;
    };

    if (all_dram_interleaved or (num_blocks_x * num_blocks_y <= num_cores_x * num_cores_y and Kt % in0_block_w == 0)) {
        CoreCoord core_range = get_core_range(num_blocks_y, num_blocks_x, num_cores_y, num_cores_x);
        bool use_mcast_1d_in0_config = is_wide or (core_range.y == 0 and mem_config.is_sharded() and
                                                   mem_config.memory_layout() == TensorMemoryLayout::WIDTH_SHARDED);
        bool use_mcast_1d_in1_config = is_tall or (core_range.y == 0 and mem_config.is_sharded() and
                                                   mem_config.memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED);
        bool use_mcast_2d_config =
            all_dram_interleaved or (core_range.y == 0 and mem_config.is_sharded() and
                                     mem_config.memory_layout() == TensorMemoryLayout::BLOCK_SHARDED);
        if (core_range.y == 1 or use_mcast_1d_in0_config) {
            return get_mcast_1d_config(
                input_tensor_a,
                input_tensor_b,
                transpose_a,
                transpose_b,
                bias_single_tile_size,
                false /* fuse_batch */,
                std::nullopt /* fused_activation */,
                true /* mcast_in0 */,
                false /* out_sharded */,
                std::nullopt /* compute_with_storage_grid_size */,
                compute_kernel_config,
                output_dtype,
                all_dram_interleaved);
        }
        if (core_range.x == 1 or use_mcast_1d_in1_config) {
            return get_mcast_1d_config(
                input_tensor_a,
                input_tensor_b,
                transpose_a,
                transpose_b,
                bias_single_tile_size,
                false /* fuse_batch */,
                std::nullopt /* fused_activation */,
                false /* mcast_in0 */,
                false /* out_sharded */,
                std::nullopt /* compute_with_storage_grid_size */,
                compute_kernel_config,
                output_dtype,
                all_dram_interleaved);
        }
        if ((core_range.y > 0 and num_blocks_x <= num_cores_x and num_blocks_y <= num_cores_y) or use_mcast_2d_config) {
            bool transpose_mcast =
                input_tensor_a.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED &&
                input_tensor_a.shard_spec().value().orientation == ShardOrientation::COL_MAJOR;
            uint32_t out_block_h = per_core_M;
            uint32_t out_block_w = per_core_N;
            out_subblock_h = 4;
            out_subblock_w = 2;
            if (out_subblock_w != per_core_N) {
                out_subblock_h = 1;
            }
            if (all_dram_interleaved) {
                in0_block_w = !transpose_mcast ? (Kt % num_cores_x == 0 ? Kt / num_cores_x : 1)
                                               : (Kt % num_cores_x == 0 ? Kt / num_cores_y : 1);
                per_core_M = !transpose_mcast ? tt::div_up(Mt, num_cores_y) : tt::div_up(Mt, num_cores_x);
                per_core_N = !transpose_mcast ? tt::div_up(Nt, num_cores_x) : tt::div_up(Nt, num_cores_y);

                auto mutlti_dim_per_core_factor = get_multi_dim_per_core_factor(
                    input_tensor_a,
                    input_tensor_b,
                    transpose_a,
                    transpose_b,
                    bias_single_tile_size,
                    per_core_M,
                    per_core_N,
                    in0_block_w,
                    utilities::estimate_interm_tile_size(compute_kernel_config, output_dtype),
                    /*adjust_in0_block_w=*/true);
                out_block_h = mutlti_dim_per_core_factor[0];
                out_block_w = mutlti_dim_per_core_factor[1];
                in0_block_w = mutlti_dim_per_core_factor[2];

                bool fp32_dest_acc_en = get_fp32_dest_acc_en(compute_kernel_config);
                auto subblock_hw =
                    bmm_op_utils::get_matmul_subblock_params(out_block_h, out_block_w, false, false, fp32_dest_acc_en);
                out_subblock_h = std::get<0>(subblock_hw);
                out_subblock_w = std::get<1>(subblock_hw);
            }
            return MatmulMultiCoreReuseMultiCastProgramConfig{
                .compute_with_storage_grid_size = {num_cores_x, num_cores_y},
                .in0_block_w = in0_block_w,
                .out_subblock_h = out_subblock_h,
                .out_subblock_w = out_subblock_w,
                .out_block_h = out_block_h,
                .out_block_w = out_block_w,
                .per_core_M = per_core_M,
                .per_core_N = per_core_N,
                .transpose_mcast = transpose_mcast,
                .fused_activation = std::nullopt,
                .fuse_batch = false,
            };
        }
    }
    return MatmulMultiCoreProgramConfig{};
}

}  // namespace ttnn::operations::matmul

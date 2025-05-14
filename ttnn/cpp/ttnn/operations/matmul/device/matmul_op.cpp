// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/matmul/device/matmul_op.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <optional>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/distributed/types.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/types.hpp"

using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal;
using ttnn::operations::unary::UnaryWithParam;

namespace {

using namespace ttnn::operations::matmul;
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

constexpr uint32_t NARROW_SHAPE_RATIO_THRESHOLD = 8;

bool is_narrow_shape(uint32_t height, uint32_t width) {
    uint32_t height_width_ratio = (height > width) ? height / width : width / height;
    return height_width_ratio > NARROW_SHAPE_RATIO_THRESHOLD || height <= ttnn::TILE_SIZE || width <= ttnn::TILE_SIZE;
}

inline bool get_fp32_dest_acc_en(const std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config) {
    bool fp32_dest_acc_en = false;
    if (compute_kernel_config) {
        std::visit(
            [&](auto&& compute_kernel_config) {
                using T = std::decay_t<decltype(compute_kernel_config)>;
                if constexpr (std::is_same_v<T, ttnn::WormholeComputeKernelConfig>) {
                    fp32_dest_acc_en = compute_kernel_config.fp32_dest_acc_en;
                }
            },
            *compute_kernel_config);
    }
    return fp32_dest_acc_en;
}

uint32_t estimate_interm_tile_size(
    const std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config,
    const tt::tt_metal::DataType output_dtype) {
    if (get_fp32_dest_acc_en(compute_kernel_config)) {
        return tt_metal::detail::TileSize(tt::DataFormat::Float32);
    }
    uint32_t result = tt_metal::detail::TileSize(tt::DataFormat::Float16_b);  // packer l1 acc
    tt::DataFormat output_data_format = tt_metal::datatype_to_dataformat_converter(output_dtype);
    uint32_t output_tile_size = tt_metal::detail::TileSize(output_data_format);
    if (output_tile_size > result) {
        result = output_tile_size;
    }
    return result;
}

bool get_broadcast_batch(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const std::optional<const MatmulProgramConfig>& matmul_program_config) {
    uint32_t batch_size_b = get_batch_size(input_tensor_b.get_padded_shape());
    bool broadcast_batch = batch_size_b == 1;
    if (!matmul_program_config.has_value()) {
        return broadcast_batch;
    }

    bool is_multi_core_reuse = std::visit(
        [](const auto& program_config) -> bool {
            using ProgramConfigType = std::decay_t<decltype(program_config)>;
            return static_cast<bool>(std::is_same_v<ProgramConfigType, MatmulMultiCoreReuseProgramConfig>);
        },
        matmul_program_config.value());
    if (is_multi_core_reuse) {
        uint32_t batch_size_a = get_batch_size(input_tensor_a.get_padded_shape());
        broadcast_batch &= batch_size_a > 1;
    }
    return broadcast_batch;
}

operation::OpPerformanceModel create_op_performance_model_for_matmul(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    const std::vector<Tensor>& output_tensors,
    const ttnn::DeviceComputeKernelConfig& compute_kernel_config) {
    const auto& in_a_shape = input_tensors.at(0).get_logical_shape();
    const auto& in_b_shape = input_tensors.at(1).get_logical_shape();
    const auto& out_shape = output_tensors.at(0).get_logical_shape();

    const auto& t = output_tensors.at(0);
    if (t.storage_type() != StorageType::DEVICE) {
        tt::log_warning(tt::LogOp, "Output tensor not on DEVICE?!");
    }

    auto arch = t.storage_type() == StorageType::DEVICE
                    ? t.device()->arch()
                    : ttnn::operations::experimental::auto_format::AutoFormat::GetDefaultDevice()->arch();
    const int num_cores = (arch == ARCH::WORMHOLE_B0) ? 8 * 8 : 9 * 12;
    const int tensix_mul_adds_per_cycle_lofi = (arch == ARCH::WORMHOLE_B0) ? 4096 : 2048;

    // Calculate number of mul/add operations
    // TODO: add bias modeling
    int64_t num_mul_adds_per_elem = in_a_shape[-1] * 2;  // 1 multiply and 1 add per element
    uint32_t batch_size = get_batch_size(out_shape);
    int64_t num_mul_adds = num_mul_adds_per_elem * out_shape[-2] * out_shape[-1] * batch_size;

    MathFidelity math_fidelity = ttnn::get_math_fidelity(compute_kernel_config);

    int ideal_dev_clock_cycles = std::ceil(
        ((float)num_mul_adds / (float)(num_cores * tensix_mul_adds_per_cycle_lofi)) *
        (float)operation::OpPerformanceModel::fidelity_multiplier(math_fidelity));

    operation::OpPerformanceModel result(input_tensors, output_tensors, ideal_dev_clock_cycles);
#if 0
    tt::log_info(tt::LogOp, "Matmul PerfModel:");
    for (auto i = 0; i < out_shape.rank() - 2; i++) {
        tt::log_info(tt::LogOp, "\t Batch Values: (Index: {}, Value: {})", i, out_shape[i]);
    }
    tt::log_info(tt::LogOp, "\t In A (H, W): ({}, {})", in_a_shape[-2], in_a_shape[-1]);
    tt::log_info(tt::LogOp, "\t In B (H, W): ({}, {})", in_b_shape[-2], in_b_shape[-1]);
    tt::log_info(tt::LogOp, "\t Out (H, W): ({}, {})", out_shape[-2], out_shape[-1]);
    tt::log_info(tt::LogOp, "\t ideal_dev_clock_cycles: {}", ideal_dev_clock_cycles);
#endif
    return result;
}

std::tuple<uint32_t, uint32_t> get_subblock_sizes(
    uint32_t m_tiles_per_core, uint32_t n_tiles_per_core, bool fp32_dest_acc_en) {
    uint32_t out_subblock_h, out_subblock_w;
    for (auto& subblock_hw : SUBBLOCK_HW_CHOICES) {
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

CoreCoord get_core_range(
    uint32_t num_blocks_rows, uint32_t num_blocks_cols, uint32_t max_num_rows, uint32_t max_num_cols) {
    CoreCoord core_range(0, 0);
    if (!(num_blocks_rows == 1 && num_blocks_cols == 1) && num_blocks_rows <= max_num_rows &&
        num_blocks_cols <= max_num_cols) {
        core_range.x = num_blocks_cols;
        core_range.y = num_blocks_rows;
    }
    return core_range;
}

inline uint32_t get_estimated_size_of_cbs(
    uint32_t per_core_M,
    uint32_t per_core_N,
    uint32_t in0_block_w,
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    uint32_t interm_single_tile_size,
    uint32_t bias_single_tile_size) {
    // Circular Buffer sizes:
    // src0   CB: per_core_M * in0_block_w * 2 (for double buffer)
    // src1   CB: per_core_N * in0_block_w * 2 (for double buffer)
    // interm CB: per_core_M * per_core_N * interm_single_tile_size
    // out    CB: per_core_M * per_core_N
    // bias   CB: per_core_M * in0_block_w
    // Ignore optional intermediate CB because not needed when need to create a
    // program config.
    tt::DataFormat in0_data_format = tt_metal::datatype_to_dataformat_converter(input_tensor_a.get_dtype());
    tt::DataFormat in1_data_format = tt_metal::datatype_to_dataformat_converter(input_tensor_b.get_dtype());
    uint32_t in0_single_tile_size = tt_metal::detail::TileSize(in0_data_format);  // use as estimate for output as well
    uint32_t in1_single_tile_size = tt_metal::detail::TileSize(in1_data_format);
    uint32_t output_single_tile_size = in0_single_tile_size;
    auto in0_buffer = input_tensor_a.buffer();
    auto in0_tile = input_tensor_a.get_tensor_spec().tile();
    uint32_t in2_block_tiles = 0;
    uint32_t in0_shard_width_in_tiles = 0;
    if (input_tensor_a.is_sharded()) {
        in0_shard_width_in_tiles = in0_buffer->shard_spec().shape()[1] / in0_tile.get_tile_shape()[1];
    }
    in2_block_tiles = per_core_M * in0_shard_width_in_tiles;

    // Calculate individual buffer sizes in bytes - use constant for buffering depth
    uint32_t in0_size = per_core_M * in0_block_w * MCAST_INPUT_BUFFERING_DEPTH * in0_single_tile_size;
    uint32_t in1_size = per_core_N * in0_block_w * MCAST_INPUT_BUFFERING_DEPTH * in1_single_tile_size;
    uint32_t out_size = per_core_M * per_core_N * output_single_tile_size;
    uint32_t in2_size = in2_block_tiles * in0_single_tile_size;
    uint32_t interm_size = per_core_M * per_core_N * interm_single_tile_size;
    uint32_t bias_size = in0_block_w * bias_single_tile_size;
    return in0_size + in1_size + out_size + interm_size + bias_size + in2_size;
}

inline uint32_t get_max_l1_space(const Tensor& input_tensor_a) {
    auto device = input_tensor_a.device();
    auto lowest_address = device->lowest_occupied_compute_l1_address();
    uint32_t max_l1_space = lowest_address.has_value() ? lowest_address.value() : device->l1_size_per_core();
    max_l1_space = max_l1_space - device->allocator()->get_base_allocator_addr(HalMemType::L1);
    return max_l1_space;
}

inline bool can_cbs_fit_in_l1(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const uint32_t bias_single_tile_size,
    uint32_t per_core_M,
    uint32_t per_core_N,
    uint32_t in0_block_w,
    const std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config,
    const tt::tt_metal::DataType output_dtype) {
    uint32_t max_l1_space = get_max_l1_space(input_tensor_a);
    uint32_t size = get_estimated_size_of_cbs(
        per_core_M,
        per_core_N,
        in0_block_w,
        input_tensor_a,
        input_tensor_b,
        estimate_interm_tile_size(compute_kernel_config, output_dtype),
        bias_single_tile_size);
    return size < max_l1_space;
}

inline uint32_t get_per_core_factor(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const uint32_t bias_single_tile_size,
    uint32_t in0_block_w,
    const std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config,
    const tt::tt_metal::DataType output_dtype) {
    uint32_t max_l1_space = get_max_l1_space(input_tensor_a);
    for (uint32_t per_core_factor = 16; per_core_factor > 1; per_core_factor /= 2) {
        uint32_t size = get_estimated_size_of_cbs(
            per_core_factor,
            per_core_factor,
            in0_block_w,
            input_tensor_a,
            input_tensor_b,
            estimate_interm_tile_size(compute_kernel_config, output_dtype),
            bias_single_tile_size);
        if (size < max_l1_space) {
            return per_core_factor;
        }
    }
    return 1;
}

inline std::vector<uint32_t> get_multi_dim_per_core_factor(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const uint32_t bias_single_tile_size,
    uint32_t per_core_M,
    uint32_t per_core_N,
    uint32_t in0_block_w,
    uint32_t interm_cb_size,
    const bool adjust_in0_block_w) {
    uint32_t max_l1_space = get_max_l1_space(input_tensor_a);
    uint32_t size = get_estimated_size_of_cbs(
        per_core_M, per_core_N, in0_block_w, input_tensor_a, input_tensor_b, interm_cb_size, bias_single_tile_size);
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
        for (auto it = factors.crbegin(); it != factors.crend(); ++it) {
            uint32_t per_core_factor_m = std::get<0>(it->second);
            uint32_t per_core_factor_n = std::get<1>(it->second);

            size = get_estimated_size_of_cbs(
                per_core_factor_m,
                per_core_factor_n,
                per_core_factor_k,
                input_tensor_a,
                input_tensor_b,
                interm_cb_size,
                bias_single_tile_size);
            if (size < max_l1_space) {
                return {per_core_factor_m, per_core_factor_n, per_core_factor_k};
            }
        }
    }
    return {1, 1, 1};
}

MatmulProgramConfig create_matmul_1d_systolic_array_program_config(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const uint32_t bias_single_tile_size,
    const CoreCoord& core_coord,
    const std::optional<const UnaryWithParam>& fused_activation,
    const bool fp32_dest_acc_en,
    const TensorMemoryLayout input_layout_a,
    const std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config,
    const tt::tt_metal::DataType output_dtype) {
    const auto& a_padded_shape = input_tensor_a.get_padded_shape();
    const auto& b_padded_shape = input_tensor_b.get_padded_shape();
    auto k_size = a_padded_shape[-1];
    auto m_size = a_padded_shape[-2];
    auto n_size = b_padded_shape[-1];
    uint32_t batch_size_a = get_batch_size(a_padded_shape);
    uint32_t batch_size_b = get_batch_size(b_padded_shape);
    bool input_b_is_batched = batch_size_b > 1;
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
        bias_single_tile_size,
        batch_and_m_tiles_per_core,
        n_tiles_per_core,
        k_tiles_per_core,
        estimate_interm_tile_size(compute_kernel_config, output_dtype),
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
    const uint32_t bias_single_tile_size,
    const bool fuse_batch,
    const std::optional<UnaryWithParam>& fused_activation,
    const bool mcast_in0,
    const bool out_sharded,
    const std::optional<const CoreCoord> compute_with_storage_grid_size,
    const std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config,
    const tt::tt_metal::DataType output_dtype,
    const bool all_dram_interleaved) {
    auto device = input_tensor_a.device();
    auto grid_size = compute_with_storage_grid_size.value_or(device->compute_with_storage_grid_size());
    uint32_t M = fuse_batch ? input_tensor_a.volume() / input_tensor_a.get_padded_shape()[-1]
                            : input_tensor_a.get_padded_shape()[-2];
    uint32_t K = input_tensor_a.get_padded_shape()[-1];
    uint32_t N = input_tensor_b.get_padded_shape()[-1];
    uint32_t per_core_M, per_core_N;
    auto in0_tile_shape = input_tensor_a.get_tensor_spec().tile().get_tile_shape();
    auto in1_tile_shape = input_tensor_b.get_tensor_spec().tile().get_tile_shape();
    if (mcast_in0) {
        per_core_M = M / in0_tile_shape[0];
        per_core_N = div_up(div_up(N, grid_size.x * grid_size.y), in1_tile_shape[1]);
    } else {
        per_core_M = div_up(div_up(M, grid_size.x * grid_size.y), in0_tile_shape[0]);
        per_core_N = N / in1_tile_shape[1];
    }
    uint32_t in0_block_w = K / in0_tile_shape[1] % 2 == 0 ? 2 : 1;
    bool per_core_N_equals_subblock_w_constraint = out_sharded && !mcast_in0;
    bool per_core_M_equals_subblock_h_constraint = out_sharded && mcast_in0;
    bool fp32_dest_acc_en = get_fp32_dest_acc_en(compute_kernel_config);

    auto mutlti_dim_per_core_factor = get_multi_dim_per_core_factor(
        input_tensor_a,
        input_tensor_b,
        bias_single_tile_size,
        per_core_M,
        per_core_N,
        in0_block_w,
        estimate_interm_tile_size(compute_kernel_config, output_dtype),
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

inline MatmulProgramConfig create_simple_matmul_program_config(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const uint32_t bias_single_tile_size,
    const std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config,
    const CoreCoord& compute_with_storage_grid_size,
    const MemoryConfig& mem_config,
    const tt::tt_metal::DataType output_dtype) {
    const auto &ashape = input_tensor_a.get_padded_shape(), bshape = input_tensor_b.get_padded_shape();
    uint32_t batch_size_a = get_batch_size(ashape);
    uint32_t num_output_tiles = batch_size_a * ashape[-2] * bshape[-1] / TILE_HW;  // Output M x N

    auto in0_tile_shape = input_tensor_a.get_tensor_spec().tile().get_tile_shape();
    auto in1_tile_shape = input_tensor_b.get_tensor_spec().tile().get_tile_shape();

    // Parameters for large matmul with reuse
    uint32_t B = batch_size_a;
    uint32_t Mt = ashape[-2] / in0_tile_shape[0];
    uint32_t Kt = ashape[-1] / in0_tile_shape[1];
    uint32_t Nt = bshape[-1] / in1_tile_shape[1];
    uint32_t in0_block_w = 2;

    TT_FATAL(input_tensor_a.storage_type() == StorageType::DEVICE, "input tensor needs to be on device");
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    uint32_t per_core_M, per_core_N, out_subblock_h, out_subblock_w;
    uint32_t num_blocks_x, num_blocks_y;

    bool all_dram_interleaved = input_tensor_a.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED &&
                                mem_config.memory_layout() == TensorMemoryLayout::INTERLEAVED &&
                                input_tensor_b.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED &&
                                input_tensor_a.memory_config().buffer_type() == BufferType::DRAM &&
                                input_tensor_b.memory_config().buffer_type() == BufferType::DRAM &&
                                mem_config.buffer_type() == BufferType::DRAM;

    uint32_t height = ashape[-2];
    uint32_t width = bshape[-1];
    bool is_narrow = is_narrow_shape(height, width);
    bool is_wide = false;
    bool is_tall = false;
    if (all_dram_interleaved && is_narrow) {
        is_wide = width > height;
        is_tall = !is_wide;
    }

    // out_subblock h/w doesn't matter
    per_core_M = get_per_core_factor(
        input_tensor_a, input_tensor_b, bias_single_tile_size, in0_block_w, compute_kernel_config, output_dtype);
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
                bias_single_tile_size,
                false /* fuse_batch */,
                std::nullopt /* fused_activation */,
                true /* mcast_in0 */,
                false /* out_sharded */,
                std::nullopt /* compute_with_storage_grid_size */,
                compute_kernel_config,
                output_dtype,
                all_dram_interleaved);
        } else if (core_range.x == 1 or use_mcast_1d_in1_config) {
            return get_mcast_1d_config(
                input_tensor_a,
                input_tensor_b,
                bias_single_tile_size,
                false /* fuse_batch */,
                std::nullopt /* fused_activation */,
                false /* mcast_in0 */,
                false /* out_sharded */,
                std::nullopt /* compute_with_storage_grid_size */,
                compute_kernel_config,
                output_dtype,
                all_dram_interleaved);
        } else if (
            (core_range.y > 0 and num_blocks_x <= num_cores_x and num_blocks_y <= num_cores_y) or use_mcast_2d_config) {
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
                    bias_single_tile_size,
                    per_core_M,
                    per_core_N,
                    in0_block_w,
                    estimate_interm_tile_size(compute_kernel_config, output_dtype),
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

MatmulProgramConfig create_matmul_program_config(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const uint32_t bias_single_tile_size,
    const std::optional<const CoreCoord> user_core_coord,
    const std::optional<UnaryWithParam>& fused_activation,
    const std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config,
    const MemoryConfig& mem_config,
    const tt::tt_metal::DataType output_dtype) {
    const auto& a_shape = input_tensor_a.get_logical_shape();
    const auto& b_shape = input_tensor_b.get_logical_shape();
    const auto& a_padded_shape = input_tensor_a.get_padded_shape();
    const auto& b_padded_shape = input_tensor_b.get_padded_shape();
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
    auto input_tensor_a_memory_config = input_tensor_a.memory_config();
    auto input_tensor_b_memory_config = input_tensor_b.memory_config();
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
                    bias_single_tile_size,
                    m_tiles_per_core,
                    n_tiles_per_core,
                    k_tiles_per_core,
                    compute_kernel_config,
                    output_dtype)) {
                return create_simple_matmul_program_config(
                    input_tensor_a,
                    input_tensor_b,
                    bias_single_tile_size,
                    compute_kernel_config,
                    core_coord,
                    mem_config,
                    output_dtype);
            }
        } else if (a_is_sharded) {
            TT_FATAL(
                a_layout != TensorMemoryLayout::WIDTH_SHARDED,
                "MatmulMultiCoreReuseProgramConfig: Cannot be width sharded");
            auto shard_shape = input_tensor_a_memory_config.shard_spec().value().shape;
            uint32_t n = b_shape[-1] / ttnn::TILE_SIZE;
            m_tiles_per_core = shard_shape[0] / ttnn::TILE_SIZE;
            n_tiles_per_core = n;
            k_tiles_per_core = shard_shape[1] / ttnn::TILE_SIZE;
        } else {
            TT_FATAL(
                input_tensor_b_memory_config.memory_layout() != TensorMemoryLayout::WIDTH_SHARDED,
                "MatmulMultiCoreReuseProgramConfig: Cannot be width sharded");
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
    auto height_width_ratio = (height > width) ? height / width : width / height;
    bool a_is_block_sharded = a_layout == TensorMemoryLayout::BLOCK_SHARDED;
    if (is_narrow_shape(height, width) || any_size_within_tile) {
        if (!a_is_block_sharded) {
            return create_matmul_1d_systolic_array_program_config(
                input_tensor_a,
                input_tensor_b,
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
                bias_single_tile_size,
                core_coord,
                fused_activation,
                fp32_dest_acc_en,
                a_layout,
                compute_kernel_config,
                output_dtype);
        }
        uint32_t k = a_shape[-1] / ttnn::TILE_SIZE;
        uint32_t n = b_shape[-1] / ttnn::TILE_SIZE;
        auto shard_shape = input_tensor_a_memory_config.shard_spec().value().shape;
        m_tiles_per_core = shard_shape[0] / ttnn::TILE_SIZE;
        n_tiles_per_core = (n * shard_shape[1]) / (k * ttnn::TILE_SIZE);
        k_tiles_per_core = std::gcd(shard_shape[1] / ttnn::TILE_SIZE, k);
    }

    n_tiles_per_core = std::max(n_tiles_per_core, (unsigned int)1);

    auto mutlti_dim_per_core_factor = get_multi_dim_per_core_factor(
        input_tensor_a,
        input_tensor_b,
        bias_single_tile_size,
        m_tiles_per_core,
        n_tiles_per_core,
        k_tiles_per_core,
        estimate_interm_tile_size(compute_kernel_config, output_dtype),
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
    const uint32_t bias_single_tile_size,
    const MemoryConfig& output_mem_config,
    const std::optional<UnaryWithParam>& fused_activation,
    const bool matmul,
    const std::optional<const CoreCoord> user_core_coord,
    const std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config,
    const tt::tt_metal::DataType output_dtype) {
    TT_FATAL(input_tensor_a.is_sharded(), "Error");
    bool fp32_dest_acc_en = get_fp32_dest_acc_en(compute_kernel_config);
    // TODO: allow overwriting of grid size by user_core_coord after allowing
    // support of arbitrary compute grid and more generic sharded output tensor
    // creation
    auto grid_size = input_tensor_a.shard_spec().value().grid.bounding_box().grid_size();

    auto in0_tile_shape = input_tensor_a.get_tensor_spec().tile().get_tile_shape();
    auto in1_tile_shape = input_tensor_b.get_tensor_spec().tile().get_tile_shape();

    // MCAST matmuls only support input_b in INTERLEAVED
    if (matmul) {
        TT_FATAL(input_tensor_b.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED, "Error");
        if ((input_tensor_a.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED or
             input_tensor_a.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED) and
            (grid_size.x > 1 or grid_size.y > 1)) {
            TT_FATAL(input_tensor_a.shard_spec().value().orientation == ShardOrientation::ROW_MAJOR, "Error");

            bool per_core_N_equals_subblock_w_constraint = false;
            if (output_mem_config.is_sharded()) {
                TT_FATAL(input_tensor_a.memory_config().buffer_type() == output_mem_config.buffer_type(), "Error");
                TT_FATAL(input_tensor_a.memory_config().memory_layout() == output_mem_config.memory_layout(), "Error");
                per_core_N_equals_subblock_w_constraint = true;
            }

            uint32_t M = input_tensor_a.volume() / input_tensor_a.get_padded_shape()[-1] / in0_tile_shape[0];
            uint32_t K = input_tensor_a.get_padded_shape()[-1] / in0_tile_shape[1];
            uint32_t N = input_tensor_b.get_padded_shape()[-1] / in1_tile_shape[1];
            auto shard_shape = input_tensor_a.shard_spec().value().shape;

            bool mcast_in0;
            uint32_t per_core_M;
            uint32_t per_core_N;
            uint32_t in0_block_w;
            if (input_tensor_a.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED) {
                mcast_in0 = true;
                per_core_M = M;
                per_core_N = div_up(N, input_tensor_a.shard_spec().value().grid.num_cores());
                in0_block_w = std::gcd(shard_shape[1] / in0_tile_shape[1], K);
            } else if (input_tensor_a.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED) {
                mcast_in0 = false;
                per_core_M = shard_shape[0] / in0_tile_shape[0];
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
                bias_single_tile_size,
                per_core_M,
                per_core_N,
                in0_block_w,
                estimate_interm_tile_size(compute_kernel_config, output_dtype),
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
        } else if (
            input_tensor_a.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED and
            (grid_size.x > 1 and grid_size.y > 1)) {
            bool transpose_mcast = input_tensor_a.shard_spec().value().orientation == ShardOrientation::COL_MAJOR;

            bool per_core_N_equals_subblock_w_constraint = false;
            if (output_mem_config.is_sharded()) {
                TT_FATAL(input_tensor_a.memory_config().buffer_type() == output_mem_config.buffer_type(), "Error");
                TT_FATAL(input_tensor_a.memory_config().memory_layout() == output_mem_config.memory_layout(), "Error");
                per_core_N_equals_subblock_w_constraint = true;
            }

            uint32_t M = input_tensor_a.volume() / input_tensor_a.get_padded_shape()[-1] / in0_tile_shape[0];
            uint32_t K = input_tensor_a.get_padded_shape()[-1] / in0_tile_shape[1];
            uint32_t N = input_tensor_b.get_padded_shape()[-1] / in1_tile_shape[1];

            auto shard_shape = input_tensor_a.shard_spec().value().shape;
            uint32_t virtual_x = transpose_mcast ? grid_size.y : grid_size.x;
            uint32_t virtual_y = transpose_mcast ? grid_size.x : grid_size.y;
            bool cores_along_x_match_grid_size = virtual_x == (K / (shard_shape[1] / in0_tile_shape[1]));
            bool cores_along_y_match_grid_size = virtual_y == (M / (shard_shape[0] / in0_tile_shape[0]));
            TT_FATAL(
                cores_along_y_match_grid_size || virtual_y == div_up(M, (shard_shape[0] / in0_tile_shape[0])),
                "Num cores along y must match provided grid size!");
            TT_FATAL(
                cores_along_x_match_grid_size || virtual_x == div_up(K, (shard_shape[1] / in0_tile_shape[1])),
                "Num cores along x must match provided grid size!");

            uint32_t per_core_M = div_up(M, virtual_y);
            uint32_t per_core_N = div_up(N, virtual_x);
            uint32_t in0_block_w = cores_along_x_match_grid_size ? std::gcd(shard_shape[1] / in0_tile_shape[1], K) : 1;

            auto mutlti_dim_per_core_factor = get_multi_dim_per_core_factor(
                input_tensor_a,
                input_tensor_b,
                bias_single_tile_size,
                per_core_M,
                per_core_N,
                in0_block_w,
                estimate_interm_tile_size(compute_kernel_config, output_dtype),
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
        TT_FATAL(input_tensor_a.memory_config().memory_layout() != TensorMemoryLayout::WIDTH_SHARDED, "Error");

        bool per_core_N_equals_subblock_w_constraint = false;
        if (output_mem_config.is_sharded()) {
            TT_FATAL(input_tensor_a.memory_config().buffer_type() == output_mem_config.buffer_type(), "Error");
            TT_FATAL(input_tensor_a.memory_config().memory_layout() == output_mem_config.memory_layout(), "Error");
            per_core_N_equals_subblock_w_constraint = true;
        }

        uint32_t M = input_tensor_a.volume() / input_tensor_a.get_padded_shape()[-1] / in0_tile_shape[0];
        uint32_t K = input_tensor_a.get_padded_shape()[-1] / in0_tile_shape[1];
        uint32_t N = input_tensor_b.get_padded_shape()[-1] / in1_tile_shape[1];

        auto in0_shard_shape = input_tensor_a.shard_spec().value().shape;
        uint32_t per_core_M = in0_shard_shape[0] / in0_tile_shape[0];
        uint32_t per_core_N = N;
        uint32_t in0_block_w = in0_shard_shape[1] / in0_tile_shape[1];

        auto subblock_hw = bmm_op_utils::get_matmul_subblock_params(
            per_core_M, per_core_N, false, per_core_N_equals_subblock_w_constraint, fp32_dest_acc_en);
        auto out_subblock_h = std::get<0>(subblock_hw);
        auto out_subblock_w = std::get<1>(subblock_hw);

        // TODO: Temporarily allow for single core; should support bcast_batch in general
        uint32_t batch_size_a = get_batch_size(input_tensor_a.get_padded_shape());
        uint32_t batch_size_b = get_batch_size(input_tensor_b.get_padded_shape());
        bool broadcast_batch = batch_size_a > 1 and batch_size_b == 1;
        TT_FATAL(!broadcast_batch, "Error");

        if (input_tensor_b.is_sharded()) {
            TT_FATAL(
                input_tensor_a.memory_config().buffer_type() == input_tensor_b.memory_config().buffer_type(), "Error");
            TT_FATAL(
                input_tensor_a.memory_config().memory_layout() == input_tensor_b.memory_config().memory_layout(),
                "Error");
            TT_FATAL(input_tensor_a.shard_spec().value().grid == input_tensor_b.shard_spec().value().grid, "Error");
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
    const uint32_t bias_single_tile_size,
    const MemoryConfig& mem_config,
    const std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<const CoreCoord> user_core_coord,
    const std::optional<UnaryWithParam>& user_fused_activation,
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
                bias_single_tile_size,
                user_core_coord,
                user_fused_activation,
                compute_kernel_config,
                mem_config,
                output_dtype);
        } else {
            tt::tt_metal::IDevice* device = input_tensor_a.device();
            auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
            return create_simple_matmul_program_config(
                input_tensor_a,
                input_tensor_b,
                bias_single_tile_size,
                compute_kernel_config,
                compute_with_storage_grid_size,
                mem_config,
                output_dtype);
        }
    } else {
        bool bmm = user_run_batched;
        return get_matmul_program_config(
            input_tensor_a,
            input_tensor_b,
            bias_single_tile_size,
            mem_config,
            std::nullopt,
            !bmm,
            user_core_coord,
            compute_kernel_config,
            output_dtype);
    }
}

inline MatmulProgramConfig get_program_config(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const uint32_t bias_single_tile_size,
    const struct Matmul* matmul) {
    if (matmul->program_config.has_value()) {
        return matmul->program_config.value();
    }
    auto config = generate_matmul_program_config(
        input_tensor_a,
        input_tensor_b,
        bias_single_tile_size,
        matmul->output_mem_config,
        matmul->compute_kernel_config,
        matmul->user_core_coord,
        matmul->user_fused_activation,
        matmul->user_run_batched,
        matmul->output_dtype.value_or(input_tensor_a.get_dtype()));
    tt::log_debug(tt::LogOp, "Auto generated program config: {}", config);

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

tt::tt_metal::Tile get_output_tile(
    const MemoryConfig& output_mem_config,
    const tt::tt_metal::Tile& in0_tile,
    const tt::tt_metal::Tile& in1_tile,
    const std::optional<const tt::tt_metal::Tile> output_tile) {
    auto in0_tile_shape = in0_tile.get_tile_shape();
    auto in1_tile_shape = in1_tile.get_tile_shape();
    if (output_tile.has_value()) {
        uint32_t in0_tile_h = in0_tile_shape[0];
        uint32_t in1_tile_w = in1_tile_shape[1];
        const auto& out_tile_shape = output_tile->get_tile_shape();
        TT_FATAL(out_tile_shape[1] > 0, "the override output tile width needs to be greater than zero");
        TT_FATAL(out_tile_shape[1] % in1_tile_w == 0, "the override output tile width be multiple of in1 tile width");
        TT_FATAL(out_tile_shape[0] > 0, "the override output tile height needs to be greater than zero");
        TT_FATAL(out_tile_shape[0] == in0_tile_h, "the override output tile height must equal to the in0 tile height");
        if (out_tile_shape[1] != in1_tile_w) {
            TT_FATAL(
                out_tile_shape[0] <= constants::FACE_HEIGHT,
                "the override output tile height must equal or less to face height");
        }
        if (!output_mem_config.is_sharded()) {
            TT_FATAL(
                out_tile_shape[1] == in1_tile_w, "the override output tile width must equal to the in0 tile width");
        }

        return output_tile.value();
    } else {
        return tt::tt_metal::Tile({in0_tile_shape[0], in1_tile_shape[1]});
    }
}

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
    for (auto& subblock_hw : SUBBLOCK_HW_CHOICES) {
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

namespace ttnn {

namespace operations {

namespace matmul {

ttnn::Shape compute_matmul_output_shape(const Tensor& input_tensor_a, const Tensor& input_tensor_b) {
    const auto input_shape_a = input_tensor_a.get_logical_shape();
    const auto input_shape_b = input_tensor_b.get_logical_shape();

    const auto a_rank = input_shape_a.rank();
    const auto b_rank = input_shape_b.rank();

    // Rank difference will be used to align batch dimensions
    const int32_t out_rank = std::max<int32_t>(a_rank, b_rank) - (a_rank == 1 || b_rank == 1);
    const int32_t rank_difference = std::max<int32_t>(0, out_rank - a_rank);

    // Initialize output shape based on the tensor with higher rank
    ttnn::Shape output_shape = (b_rank > a_rank) ? input_shape_b : input_shape_a;

    // Handle batch dimensions for the case where b_rank > a_rank
    for (auto index = 0; index < rank_difference; ++index) {
        TT_FATAL(input_shape_b[index] == 1, "When in1 rank greater than in0 rank front dimensions need to be 1");
        output_shape[index] = input_shape_b[index];
    }

    // Copy dimensions from input_shape_a except the last one
    for (auto index = 0; index < a_rank - 1; ++index) {
        output_shape[rank_difference + index] = input_shape_a[index];
    }

    // The last dimension comes from input_tensor_b
    output_shape[-1] = input_shape_b[-1];

    // Handle the vector matmul case: if a_rank == 1, remove the second-to-last dimension
    if (a_rank == 1 && output_shape.rank() > 1) [[unlikely]] {
        ttnn::SmallVector<uint32_t> new_shape(output_shape.rank() - 1);
        // Copy all elements except the second-to-last dimension
        size_t dst_idx = 0;
        for (size_t src_idx = 0; src_idx < output_shape.rank(); ++src_idx) {
            if (src_idx != output_shape.rank() - 2) {
                new_shape[dst_idx++] = output_shape[src_idx];
            }
        }
        output_shape = ttnn::Shape(new_shape);
    }

    // Handle the case where b_rank == 1, remove the last dimension
    if (b_rank == 1) [[unlikely]] {
        ttnn::SmallVector<uint32_t> new_shape(output_shape.rank() - 1);
        for (auto index = 0; index < output_shape.rank() - 1; ++index) {
            new_shape[index] = output_shape[index];
        }
        output_shape = ttnn::Shape(new_shape);
    }

    return output_shape;
}

Matmul create_matmul_struct(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const struct Matmul& parameters,
    const std::vector<std::optional<Tensor>>& optional_output_tensors) {
    auto arch = input_tensor_a.device()->arch();
    const bool has_user_grid = parameters.user_core_coord.has_value();
    const bool has_program_config = parameters.program_config.has_value();
    bool are_inputs_low_precision_df =
        ((input_tensor_a.get_dtype() == DataType::BFLOAT8_B || input_tensor_a.get_dtype() == DataType::BFLOAT4_B) &&
         (input_tensor_b.get_dtype() == DataType::BFLOAT8_B || input_tensor_b.get_dtype() == DataType::BFLOAT4_B));
    const auto increase_fidelity = !has_program_config && !has_user_grid && !are_inputs_low_precision_df;
    auto math_fidelity = increase_fidelity ? MathFidelity::HiFi2 : MathFidelity::LoFi;
    bool are_inputs_32F =
        (input_tensor_a.get_dtype() == DataType::FLOAT32 && input_tensor_b.get_dtype() == DataType::FLOAT32);
    math_fidelity = are_inputs_32F ? MathFidelity::HiFi4 : math_fidelity;

    bool broadcast_batch =
        parameters.bcast_batch.value_or(get_broadcast_batch(input_tensor_a, input_tensor_b, parameters.program_config));
    TT_FATAL(!(has_user_grid && has_program_config), "Cannot use both user core grid/coordinates and a program config");

    const bool is_optional_output_tensor =
        !optional_output_tensors.empty() && optional_output_tensors.at(0).has_value();
    std::optional<DataType> output_dtype = parameters.output_dtype;
    MemoryConfig output_mem_config = parameters.output_mem_config;

    if (is_optional_output_tensor) {
        const auto& optional_output_tensor = optional_output_tensors.at(0);
        if (output_mem_config == tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG) {
            output_mem_config = optional_output_tensor->memory_config();
        } else {
            TT_FATAL(
                optional_output_tensor->memory_config() == output_mem_config,
                "Memory config mismatch between optional output tensor {} & "
                "output tensor {}",
                optional_output_tensor->memory_config(),
                output_mem_config);
        }

        if (output_dtype.has_value()) {
            TT_FATAL(
                optional_output_tensor->get_dtype() == output_dtype.value(),
                "Type mismatch between optional output tensor {} & output tensor {}",
                optional_output_tensor->get_dtype(),
                output_dtype.value());
        } else {
            output_dtype = optional_output_tensor->get_dtype();
        }
    } else {
        if (!output_dtype.has_value()) {
            output_dtype = input_tensor_a.get_dtype();
        }
    }
    bool is_float_32 = output_dtype == DataType::FLOAT32;
    auto kernel_config_val = init_device_compute_kernel_config(
        arch,
        parameters.compute_kernel_config,
        math_fidelity,
        /*default_approx_mode=*/false,
        /*default_fp32_acc=*/is_float_32,
        /*default_l1_acc=*/!is_float_32);
    auto in0_tile = input_tensor_a.get_tensor_spec().tile();
    auto in1_tile = input_tensor_b.get_tensor_spec().tile();
    tt::tt_metal::Tile output_tile = get_output_tile(output_mem_config, in0_tile, in1_tile, parameters.output_tile);

    return Matmul{
        parameters.program_config,
        broadcast_batch,
        output_mem_config,
        output_dtype,
        kernel_config_val,
        parameters.untilize_out,
        parameters.user_core_coord,
        parameters.user_fused_activation,
        parameters.user_run_batched,
        parameters.transpose_a,
        parameters.transpose_b,
        output_tile,
        parameters.global_cb,
        parameters.sub_device_id};
}

Tensor matmul(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const std::optional<const Tensor>& bias,
    const struct Matmul& parameters,
    const QueueId queue_id,
    const std::optional<Tensor>& optional_output_tensor) {
    std::vector<std::optional<const Tensor>> optional_input_tensors = {};
    if (bias.has_value()) {
        optional_input_tensors.push_back(bias.value());
    } else {
        optional_input_tensors.push_back(std::nullopt);
    }

    return operation::run(
               create_matmul_struct(input_tensor_a, input_tensor_b, parameters, {optional_output_tensor}),
               {input_tensor_a, input_tensor_b},
               optional_input_tensors,
               {optional_output_tensor},
               queue_id)
        .at(0);
}

std::vector<Tensor> matmul_batched_weights(
    const Tensor& input_tensor_a,
    const std::vector<Tensor>& input_tensors_b,
    const std::optional<const Tensor>& bias,
    const struct Matmul& parameters,
    const QueueId queue_id,
    const std::optional<Tensor>& optional_output_tensor) {
    std::vector<std::optional<const Tensor>> optional_input_tensors = {};
    if (bias.has_value()) {
        optional_input_tensors.push_back(bias.value());
    } else {
        optional_input_tensors.push_back(std::nullopt);
    }

    std::vector<Tensor> input_tensors = input_tensors_b;
    input_tensors.insert(input_tensors.begin(), input_tensor_a);

    return operation::run(
        create_matmul_struct(input_tensor_a, input_tensors_b[0], parameters, {optional_output_tensor}),
        input_tensors,
        optional_input_tensors,
        {optional_output_tensor},
        queue_id);
}

void check_tensor_in_grid(const Tensor& tensor, const CoreCoord& grid_size) {
    // Validate tensor is within grid if sharded and not in DRAM
    if (tensor.memory_config().is_sharded() && tensor.memory_config().buffer_type() != BufferType::DRAM) {
        const auto& shard_spec = tensor.memory_config().shard_spec().value();
        const auto& shard_grid = shard_spec.grid;
        CoreRange range(CoreCoord(0, 0), grid_size);
        TT_FATAL(
            range.contains(shard_grid),
            "Tensor shard spec grid must be within config grid! Shard grid: {}, Config grid: {}",
            shard_grid,
            range);
    }
}

void Matmul::validate(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    const std::vector<std::optional<Tensor>>& optional_output_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);
    const auto& a_shape = input_tensor_a.get_logical_shape();
    const auto& b_shape = input_tensor_b.get_logical_shape();
    const auto& b_shape_aligned = input_tensor_b.get_padded_shape();
    auto in0_tile_shape = input_tensor_a.get_tensor_spec().tile().get_tile_shape();
    auto in1_tile_shape = input_tensor_b.get_tensor_spec().tile().get_tile_shape();

    if (input_tensor_a.device()->arch() == tt::ARCH::GRAYSKULL) {
        TT_FATAL(
            (in0_tile_shape[1] == TILE_WIDTH && in0_tile_shape[0] == TILE_HEIGHT),
            "Grayskull does not support tiny tile");
        TT_FATAL(
            (in1_tile_shape[1] == TILE_WIDTH && in1_tile_shape[0] == TILE_HEIGHT),
            "Grayskull does not support tiny tile");
    }

    TT_FATAL(
        (in0_tile_shape[1] == TILE_WIDTH && in1_tile_shape[0] == TILE_WIDTH),
        "Input tile dims must have inner dim equal to 32 due to llk constraints");

    TT_FATAL(
        (input_tensor_a.get_layout() == Layout::TILE && input_tensor_b.get_layout() == Layout::TILE),
        "Inputs to matmul must be tilized");
    TT_FATAL(
        a_shape[-1] == b_shape[-2],
        "The width of the first tensor must be equal to the height of the "
        "second tensor. Mismatch: width={} height={}",
        a_shape[-1],
        b_shape[-2]);

    const bool is_optional_output_tensor =
        !optional_output_tensors.empty() && optional_output_tensors.at(0).has_value();

    TT_FATAL(optional_input_tensors.size() == 1, "Error");

    const auto output_tensor_spec = this->compute_output_specs(input_tensors, {}, optional_input_tensors).at(0);
    if (is_optional_output_tensor) {
        const auto& optional_output_tensor_c = optional_output_tensors.at(0);
        const auto& optional_output_tensor_shape = optional_output_tensor_c->get_logical_shape();
        TT_FATAL(
            optional_output_tensor_shape == output_tensor_spec.logical_shape(),
            "Shape of Optional Output Tensor {} doesnt match Output Tensor {}",
            optional_output_tensor_shape,
            output_tensor_spec.logical_shape());
        TT_FATAL(
            optional_output_tensor_c->get_dtype() == this->output_dtype.value(),
            "Type mismatch between optional output tensor {} & output tensor {}",
            optional_output_tensor_c->get_dtype(),
            this->output_dtype.value());
        TT_FATAL(
            optional_output_tensor_c->memory_config() == this->output_mem_config,
            "Memory config mismatch between optional output tensor {} & output "
            "tensor {}",
            optional_output_tensor_c->memory_config(),
            this->output_mem_config);
    } else {
        TT_FATAL(
            output_tensor_spec.memory_config().memory_layout() == this->output_mem_config.memory_layout(),
            "Mismatch between computed {} and provided {} mem config memory layout",
            output_tensor_spec.memory_config().memory_layout(),
            this->output_mem_config.memory_layout());
        TT_FATAL(
            output_tensor_spec.memory_config().buffer_type() == this->output_mem_config.buffer_type(),
            "Mismatch between computed {} and provided {} mem config buffer type",
            output_tensor_spec.memory_config().buffer_type(),
            this->output_mem_config.buffer_type());
        if (this->output_mem_config.shard_spec().has_value() &&
            output_tensor_spec.memory_config() != this->output_mem_config) {
            log_warning(
                tt::LogOp,
                "Mismatch between computed {} and provided {} mem config. Using computed config.",
                output_tensor_spec.memory_config(),
                this->output_mem_config);
        }
    }

    TT_FATAL(this->bcast_batch.has_value(), "Error: bcast_batch field should have been automatically populated");
    TT_FATAL(this->output_tile.has_value(), "Error: output_tile field should have been automatically populated");
    if (this->bcast_batch.value()) {
        TT_FATAL(
            get_batch_size(b_shape) == 1,
            "matmul (batch bcast variant) expects input tensors of shapes "
            "BCMK*11KN=BCMN or equivalent");
    } else {
        // same condition as above, different message
        TT_FATAL(
            a_shape.rank() == b_shape.rank() && "bmm (non-bcast matmul) expects input tensors of the same rank",
            "Error");
        for (auto i = 0; i < a_shape.rank() - 2; i++) {
            TT_FATAL(
                a_shape[i] == b_shape[i],
                "bmm (non-bcast matmul) expects input tensors of shapes "
                "BCMK*BCKN=BCMN or equivalent");
        }
    }

    TT_FATAL(is_floating_point(input_tensor_a.get_dtype()), "Unsupported data format");
    TT_FATAL(
        input_tensor_a.storage_type() == StorageType::DEVICE and input_tensor_b.storage_type() == StorageType::DEVICE,
        "Operands to matmul need to be on device!");
    TT_FATAL(
        input_tensor_a.buffer() != nullptr and input_tensor_b.buffer() != nullptr,
        "Operands to matmul need to be allocated in buffers on device!");
    TT_FATAL(input_tensor_a.device() == input_tensor_b.device(), "Operands to matmul need to be on the same device!");

    const auto& optional_bias = optional_input_tensors.at(0);
    uint32_t bias_single_tile_size = 0;
    if (optional_bias.has_value()) {
        auto bias_data_format = tt_metal::datatype_to_dataformat_converter(optional_bias.value().get_dtype());
        bias_single_tile_size = tt_metal::detail::TileSize(bias_data_format);
    }
    MatmulProgramConfig chosen_program_config =
        get_program_config(input_tensor_a, input_tensor_b, bias_single_tile_size, this);

    if (std::holds_alternative<MatmulMultiCoreReuseMultiCast1DProgramConfig>(chosen_program_config) &&
        this->global_cb.has_value() && input_tensor_b.is_sharded() && input_tensor_b.buffer()->is_dram()) {
        for (uint32_t i = 1; i < input_tensors.size(); ++i) {
            TT_FATAL(
                input_tensor_b.get_logical_shape() == input_tensors[i].get_logical_shape(),
                "for multi-tensor matmul, all weight tensors must have the same logical_shape, {} is not equal to {}",
                input_tensor_b.get_logical_shape(),
                input_tensors[i].get_logical_shape());
            TT_FATAL(
                input_tensor_b.get_padded_shape() == input_tensors[i].get_padded_shape(),
                "for multi-tensor matmul, all weight tensors must have the same padded_shape {} is not equal to {}",
                input_tensor_b.get_padded_shape(),
                input_tensors[i].get_padded_shape());
            TT_FATAL(
                input_tensor_b.get_tensor_spec() == input_tensors[i].get_tensor_spec(),
                "for multi-tensor matmul, all weight tensors must have the same tensor_spec {} is not equal to {}",
                input_tensor_b.get_tensor_spec(),
                input_tensors[i].get_tensor_spec());
            TT_FATAL(
                input_tensor_b.get_layout() == input_tensors[i].get_layout(),
                "for multi-tensor matmul, all weight tensors must have the same layout {} is not equal to {}",
                input_tensor_b.get_layout(),
                input_tensors[i].get_layout());
            TT_FATAL(
                input_tensor_b.get_dtype() == input_tensors[i].get_dtype(),
                "for multi-tensor matmul, all weight tensors must have the same _dtype {} is not equal to {}",
                input_tensor_b.get_dtype(),
                input_tensors[i].get_dtype());
        }
    } else {
        TT_FATAL(input_tensors.size() == 2, "Error");
    }

    if (optional_bias.has_value()) {
        const auto& bias = optional_bias.value();
        auto bias_tile_shape = bias.tensor_spec().tile().get_tile_shape();
        TT_FATAL(
            (bias_tile_shape[0] == in0_tile_shape[0] && bias_tile_shape[1] == in1_tile_shape[1]),
            "Input tile dims must have inner dim equal to 32 due to llk "
            "constraints");
        TT_FATAL(bias.get_layout() == Layout::TILE, "Unsupported input layout");
        const auto& bias_shape = bias.get_logical_shape();
        const auto& bias_shape_aligned = bias.get_padded_shape();
        uint32_t bias_batch_size = get_batch_size(bias_shape);
        TT_FATAL(bias_batch_size == 1, "Unsupported bias shape: batch size not equal to 1.");
        TT_FATAL(
            bias_shape_aligned[-2] == in0_tile_shape[0],
            "Unsupported bias shape: padded second last dimension of bias, "
            "{}, not equal to tile height, {}",
            bias_shape_aligned[-2],
            in0_tile_shape[0]);
        TT_FATAL(
            bias_shape_aligned[-1] == b_shape_aligned[-1],
            "Unsupported bias shape: padded last dimension of bias, {}, not "
            "equal to second input's padded last "
            "dimension, {}.",
            bias_shape_aligned[-1],
            b_shape_aligned[-1]);
        TT_FATAL(
            bias_shape[-1] >= b_shape[-1],
            "Unsupported bias shape: last dimension of bias, {}, not equal to "
            "or greater than second input's last "
            "dimension, {}.",
            bias_shape[-1],
            b_shape[-1]);
    }

    if (this->untilize_out) {
        TT_FATAL(this->output_dtype.has_value(), "Error");
        TT_FATAL(
            (this->output_dtype.value() == DataType::BFLOAT16) || (this->output_dtype.value() == DataType::FLOAT32),
            "Unsupported data type: {}",
            this->output_dtype.value());
    }

    std::visit(
        [input_tensor_a, input_tensor_b, optional_bias, in0_tile_shape, in1_tile_shape, this](
            const auto& program_config) {
            using ProgramConfigType = std::decay_t<decltype(program_config)>;
            // TODO: For 1D and 2D mcasts, we don't check if tensor is single core
            // or single row/col We can uplift these variants to skip mcasting to
            // support single core (1D) or single row/col (2D)
            if constexpr (std::is_same_v<ProgramConfigType, MatmulMultiCoreReuseMultiCast1DProgramConfig>) {
                TT_FATAL(
                    program_config.per_core_M % program_config.out_block_h == 0,
                    "Error: incompatible values {} and {}",
                    program_config.per_core_M,
                    program_config.out_block_h);
                TT_FATAL(
                    program_config.per_core_N % program_config.out_block_w == 0,
                    "Error: incompatible values {} and {}",
                    program_config.per_core_N,
                    program_config.out_block_w);
                TT_FATAL(
                    program_config.out_block_h % program_config.out_subblock_h == 0,
                    "Error: incompatible values {} and {}",
                    program_config.out_block_h,
                    program_config.out_subblock_h);
                TT_FATAL(
                    program_config.out_block_w % program_config.out_subblock_w == 0,
                    "Error: incompatible values {} and {}",
                    program_config.out_block_w,
                    program_config.out_subblock_w);
                TT_FATAL(
                    !(program_config.mcast_in0 && program_config.gather_in0),
                    "Matmul1D does not support mcast_in0 and gather_in0 at the "
                    "same time.");

                // Gather in0 specific validation
                if (program_config.gather_in0) {
                    TT_FATAL(
                        program_config.num_global_cb_receivers > 0, "Num global CB receivers must be greater than 0.");
                    TT_FATAL(
                        input_tensor_a.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED,
                        "Input tensor A must be width sharded when using gather_in0.");
                    TT_FATAL(
                        input_tensor_b.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED ||
                            (input_tensor_b.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED &&
                             input_tensor_b.buffer()->buffer_type() == tt_metal::BufferType::DRAM),
                        "Input tensor B must be width sharded or DRAM interleaved when using gather_in0.");
                    if (!this->global_cb.has_value() && input_tensor_b.is_sharded()) {
                        if (input_tensor_b.buffer()->buffer_type() == tt_metal::BufferType::L1) {
                            TT_FATAL(
                                input_tensor_a.shard_spec().value().grid == input_tensor_b.shard_spec().value().grid,
                                "Input tensor A and B must be sharded on the same cores "
                                "when using gather_in0.");
                        }
                    }
                    TT_FATAL(
                        this->output_mem_config.is_sharded(), "Output tensor must be sharded when using gather_in0.");
                    TT_FATAL(
                        this->output_mem_config.shard_spec().has_value(),
                        "Output shard spec must be provided when using gather_in0.");

                    if (!input_tensor_b.is_sharded()) {
                        TT_FATAL(
                            !this->global_cb.has_value(),
                            "Global CB is not supported for DRAM_INTERLEAVED in1 when using gather_in0.");
                        TT_FATAL(
                            input_tensor_b.get_layout() == Layout::TILE,
                            "Input tensor B must be TILE_LAYOUT when DRAM_INTERLEAVED when using gather_in0.");
                        TT_FATAL(
                            input_tensor_a.shard_spec().value().grid ==
                                this->output_mem_config.shard_spec().value().grid,
                            "Input tensor A and output tensor must be sharded on the same cores when using gather_in0 "
                            "and in1 is DRAM_INTERLEAVED.");
                    }

                    if (!this->global_cb.has_value()) {
                        TT_FATAL(
                            program_config.num_global_cb_receivers == 1,
                            "Num global CB receivers must be 1 when global CB is not provided.");
                    }

                    TT_FATAL(!optional_bias.has_value(), "Bias is not supported when using gather_in0.");
                } else {
                    // Checks specific to non-gather configs
                    check_tensor_in_grid(input_tensor_a, program_config.compute_with_storage_grid_size);
                    check_tensor_in_grid(input_tensor_b, program_config.compute_with_storage_grid_size);
                }
                if (program_config.mcast_in0 || program_config.gather_in0) {
                    if (input_tensor_a.is_sharded()) {
                        TT_FATAL(program_config.fuse_batch, "Error: Batch fusion must be enabled.");
                        TT_FATAL(
                            input_tensor_a.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED,
                            "Error: input_tensor_a must be width sharded. Provided tensor memory layout: {}",
                            input_tensor_a.memory_config().memory_layout());
                        if (this->output_mem_config.is_sharded()) {
                            TT_FATAL(
                                input_tensor_a.memory_config().buffer_type() == this->output_mem_config.buffer_type(),
                                "Error: Buffer type mismatch.");
                            TT_FATAL(
                                input_tensor_a.memory_config().memory_layout() ==
                                    this->output_mem_config.memory_layout(),
                                "Error: Memory layout mismatch.");
                        }
                        TT_FATAL(
                            input_tensor_a.shard_spec().value().orientation == ShardOrientation::ROW_MAJOR,
                            "Error: Shard orientation must be ROW_MAJOR.");
                        uint32_t M =
                            (program_config.fuse_batch ? input_tensor_a.volume() / input_tensor_a.get_padded_shape()[-1]
                                                       : input_tensor_a.get_padded_shape()[-2]) /
                            in0_tile_shape[0];
                        uint32_t N = input_tensor_b.get_padded_shape()[-1] / in1_tile_shape[1];
                        uint32_t K = input_tensor_a.get_padded_shape()[-1] / in0_tile_shape[1];
                        uint32_t per_core_M = program_config.per_core_M;
                        uint32_t per_core_N = program_config.per_core_N;
                        auto shard_shape = input_tensor_a.shard_spec().value().shape;

                        // No padding
                        TT_FATAL(M == per_core_M, "Error: M ({}) must be equal to per_core_M ({}).", M, per_core_M);
                        TT_FATAL(
                            per_core_M == (shard_shape[0] / in0_tile_shape[0]),
                            "Error: per_core_M must be equal to shard_shape[0] ({}) / in0_tile_shape[0] ({}).",
                            shard_shape[0],
                            in0_tile_shape[0]);
                        TT_FATAL(
                            K % program_config.in0_block_w == 0,
                            "Error: K {} must be divisible by in0_block_w {}.",
                            K,
                            program_config.in0_block_w);
                        if (!program_config.gather_in0) {  // Padding allowed for gather_in0
                            TT_FATAL(
                            (shard_shape[1] / in0_tile_shape[1]) % program_config.in0_block_w == 0,
                            "Error: shard_shape[1] ({}) / in0_tile_shape[1] ({}) must be divisible by in0_block_w.",
                            shard_shape[1],
                            in0_tile_shape[1]);
                        }
                    }
                    if (this->output_mem_config.is_sharded()) {
                        TT_FATAL(
                            this->output_mem_config.memory_layout() == TensorMemoryLayout::WIDTH_SHARDED,
                            "Error: Output memory layout must be WIDTH_SHARDED. Provided tensor memory layout: {}",
                            this->output_mem_config.memory_layout());
                        uint32_t M =
                            (program_config.fuse_batch ? input_tensor_a.volume() / input_tensor_a.get_padded_shape()[-1]
                                                       : input_tensor_a.get_padded_shape()[-2]) /
                            in0_tile_shape[0];
                        uint32_t N = input_tensor_b.get_padded_shape()[-1] / in1_tile_shape[1];
                        uint32_t per_core_M = program_config.per_core_M;
                        uint32_t per_core_N = program_config.per_core_N;

                        // No padding
                        TT_FATAL(M == per_core_M, "Error: M {} must be equal to per_core_M {}.", M, per_core_M);

                        TT_FATAL(
                            program_config.out_subblock_w == per_core_N || program_config.out_subblock_h == 1,
                            "Error: out_subblock_w must be equal to per_core_N or out_subblock_h must be equal to 1.");
                        TT_FATAL(
                            program_config.out_block_w == per_core_N || program_config.out_block_h == 1,
                            "Error: out_block_w must be equal to per_core_N or out_block_h must be equal to 1.");
                    }
                    if (input_tensor_b.buffer()->buffer_type() == tt_metal::BufferType::L1 &&
                        input_tensor_b.memory_config().is_sharded()) {
                        TT_FATAL(
                            input_tensor_b.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED,
                            "Operand B can only be interleaved or L1 width sharded.");
                        TT_FATAL(
                            program_config.per_core_N ==
                                (input_tensor_b.shard_spec().value().shape[1] / in1_tile_shape[1]),
                            "Shard width must match per core N.");
                        if (optional_bias.has_value()) {
                            TT_FATAL(
                                input_tensor_b.shard_spec().value().shape[1] ==
                                    optional_bias.value().shard_spec().value().shape[1],
                                "Bias shard spec width must match second inputs shard spec "
                                "width.");
                        }
                    }
                } else {
                    if (input_tensor_a.memory_config().is_sharded()) {
                        TT_FATAL(program_config.fuse_batch, "Error: Batch fusion must be enabled.");
                        TT_FATAL(
                            input_tensor_a.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED,
                            "Error: input_tensor_a must be height sharded.");
                        if (this->output_mem_config.is_sharded()) {
                            TT_FATAL(
                                input_tensor_a.memory_config().buffer_type() == this->output_mem_config.buffer_type(),
                                "Error: Buffer type mismatch.");
                            TT_FATAL(
                                input_tensor_a.memory_config().memory_layout() ==
                                    this->output_mem_config.memory_layout(),
                                "Error: Memory layout mismatch.");
                        }
                        TT_FATAL(
                            input_tensor_a.shard_spec().value().orientation == ShardOrientation::ROW_MAJOR,
                            "Error: Shard orientation must be ROW_MAJOR.");
                        uint32_t M =
                            (program_config.fuse_batch ? input_tensor_a.volume() / input_tensor_a.get_padded_shape()[-1]
                                                       : input_tensor_a.get_padded_shape()[-2]) /
                            in0_tile_shape[0];
                        uint32_t K = input_tensor_a.get_padded_shape()[-1] / in0_tile_shape[1];
                        uint32_t per_core_M = program_config.per_core_M;
                        auto shard_shape = input_tensor_a.shard_spec().value().shape;
                        TT_FATAL(
                            div_up(M, per_core_M) <= input_tensor_a.shard_spec().value().grid.num_cores(),
                            "Error: M must be divisible by per_core_M.");
                        TT_FATAL(
                            per_core_M == (shard_shape[0] / in0_tile_shape[0]),
                            "Error: per_core_M must be equal to shard_shape[0] / in0_tile_shape[0].");
                        TT_FATAL(K % program_config.in0_block_w == 0, "Error: K must be divisible by in0_block_w.");
                        TT_FATAL(
                            K == (shard_shape[1] / in0_tile_shape[1]),
                            "Error: K must be equal to shard_shape[1] / in0_tile_shape[1].");
                    }
                    if (this->output_mem_config.is_sharded()) {
                        TT_FATAL(
                            this->output_mem_config.memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED,
                            "Error: Output memory layout must be HEIGHT_SHARDED.");
                        uint32_t M =
                            (program_config.fuse_batch ? input_tensor_a.volume() / input_tensor_a.get_padded_shape()[-1]
                                                       : input_tensor_a.get_padded_shape()[-2]) /
                            in0_tile_shape[0];
                        uint32_t N = input_tensor_b.get_padded_shape()[-1] / in1_tile_shape[1];
                        uint32_t per_core_M = program_config.per_core_M;
                        uint32_t per_core_N = program_config.per_core_N;

                        TT_FATAL(N == per_core_N, "Error: N must be equal to per_core_N.");
                        TT_FATAL(
                            program_config.out_subblock_w == per_core_N || program_config.out_subblock_h == 1,
                            "Error: out_subblock_w must be equal to per_core_N or out_subblock_h must be equal to 1.");
                        TT_FATAL(
                            program_config.out_block_w == per_core_N || program_config.out_block_h == 1,
                            "Error: out_block_w must be equal to per_core_N or out_block_h must be equal to 1.");
                    }
                    TT_FATAL(
                        input_tensor_b.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
                        "Error: Operand B must be interleaved.");
                }
            } else if constexpr (std::is_same_v<
                                     ProgramConfigType,
                                     MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig>) {
                TT_FATAL(input_tensor_a.is_sharded(), "Error");
                TT_FATAL(this->output_mem_config.is_sharded(), "Error");
                TT_FATAL(input_tensor_a.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED, "Error");
                TT_FATAL(
                    input_tensor_a.memory_config().buffer_type() == this->output_mem_config.buffer_type(), "Error");
                TT_FATAL(
                    input_tensor_a.memory_config().memory_layout() == this->output_mem_config.memory_layout(), "Error");
                TT_FATAL(input_tensor_a.shard_spec().value().orientation == ShardOrientation::ROW_MAJOR, "Error");
                uint32_t M = input_tensor_a.volume() / input_tensor_a.get_padded_shape()[-1] / in0_tile_shape[0];
                uint32_t N = input_tensor_b.get_padded_shape()[-1] / in1_tile_shape[1];
                uint32_t K = input_tensor_a.get_padded_shape()[-1] / in0_tile_shape[1];
                uint32_t per_core_M = program_config.per_core_M;
                uint32_t per_core_N = program_config.per_core_N;
                auto shard_shape = input_tensor_a.shard_spec().value().shape;

                // No padding
                TT_FATAL(M == per_core_M, "Error");
                TT_FATAL(M == 1, "currently only support in0 tensor height of tile height");
                TT_FATAL(per_core_M == (shard_shape[0] / in0_tile_shape[0]), "Error");
                TT_FATAL(K % program_config.in0_block_w == 0, "Error");
                TT_FATAL((shard_shape[1] / in0_tile_shape[1]) % program_config.in0_block_w == 0, "Error");

                // tensor in1
                TT_FATAL(input_tensor_b.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED, "Error");
            } else if constexpr (std::is_same_v<ProgramConfigType, MatmulMultiCoreReuseMultiCastProgramConfig>) {
                check_tensor_in_grid(input_tensor_a, program_config.compute_with_storage_grid_size);
                check_tensor_in_grid(input_tensor_b, program_config.compute_with_storage_grid_size);
                TT_FATAL(
                    program_config.per_core_M % program_config.out_block_h == 0,
                    "Error: incompatible values {} and {}",
                    program_config.per_core_M,
                    program_config.out_block_h);
                TT_FATAL(
                    program_config.per_core_N % program_config.out_block_w == 0,
                    "Error: incompatible values {} and {}",
                    program_config.per_core_N,
                    program_config.out_block_w);
                TT_FATAL(
                    program_config.out_block_h % program_config.out_subblock_h == 0,
                    "Error: incompatible values {} and {}",
                    program_config.out_block_h,
                    program_config.out_subblock_h);
                TT_FATAL(
                    program_config.out_block_w % program_config.out_subblock_w == 0,
                    "Error: incompatible values {} and {}",
                    program_config.out_block_w,
                    program_config.out_subblock_w);
                if (input_tensor_a.memory_config().is_sharded()) {
                    TT_FATAL(program_config.fuse_batch, "Error");
                    auto tensor_a_memory_layout = input_tensor_a.memory_config().memory_layout();
                    uint32_t M = input_tensor_a.volume() / input_tensor_a.get_padded_shape()[-1] / in0_tile_shape[0];
                    uint32_t K = input_tensor_a.get_padded_shape()[-1] / in0_tile_shape[1];
                    uint32_t N = input_tensor_b.get_padded_shape()[-1] / in1_tile_shape[1];
                    uint32_t per_core_M = program_config.per_core_M;
                    auto shard_shape = input_tensor_a.shard_spec().value().shape;

                    TT_FATAL(
                        tensor_a_memory_layout == TensorMemoryLayout::BLOCK_SHARDED ||
                            tensor_a_memory_layout == TensorMemoryLayout::HEIGHT_SHARDED,
                        "Unsupported memory layout {}.",
                        tensor_a_memory_layout);

                    if (tensor_a_memory_layout == TensorMemoryLayout::BLOCK_SHARDED) {
                        if (program_config.transpose_mcast) {
                            TT_FATAL(
                                input_tensor_a.shard_spec().value().orientation == ShardOrientation::COL_MAJOR,
                                "Error");
                        } else {
                            TT_FATAL(
                                input_tensor_a.shard_spec().value().orientation == ShardOrientation::ROW_MAJOR,
                                "Error");
                        }
                        if (this->output_mem_config.is_sharded()) {
                            TT_FATAL(
                                input_tensor_a.memory_config().buffer_type() == this->output_mem_config.buffer_type(),
                                "Error");
                            TT_FATAL(
                                input_tensor_a.memory_config().memory_layout() ==
                                    this->output_mem_config.memory_layout(),
                                "Error");
                        }

                    } else if (tensor_a_memory_layout == TensorMemoryLayout::HEIGHT_SHARDED) {
                        TT_FATAL(!program_config.transpose_mcast, "Error");
                        TT_FATAL(K == program_config.in0_block_w, "Error");
                        TT_FATAL(program_config.in0_block_w == (shard_shape[1] / in0_tile_shape[1]), "Error");
                        TT_FATAL(
                            input_tensor_a.shard_spec()->grid.bounding_box().start_coord.x ==
                                input_tensor_a.shard_spec()->grid.bounding_box().end_coord.x,
                            "Error");
                    }

                    TT_FATAL(per_core_M == (shard_shape[0] / in0_tile_shape[0]), "Error");
                    TT_FATAL((shard_shape[1] / in0_tile_shape[1]) % program_config.in0_block_w == 0, "Error");
                }

                if (input_tensor_b.memory_config().is_sharded()) {
                    TT_FATAL(!program_config.transpose_mcast, "Error");
                    auto tensor_b_memory_layout = input_tensor_b.memory_config().memory_layout();
                    TT_FATAL(tensor_b_memory_layout == TensorMemoryLayout::WIDTH_SHARDED, "Error");
                    if (input_tensor_b.buffer()->buffer_type() != tt_metal::BufferType::DRAM) {
                        const auto tensor_a_memory_layout = input_tensor_a.memory_config().memory_layout();
                        TT_FATAL(
                            (input_tensor_a.memory_config().is_sharded() &&
                             tensor_a_memory_layout == TensorMemoryLayout::HEIGHT_SHARDED) ||
                                tensor_a_memory_layout == TensorMemoryLayout::INTERLEAVED,
                            "Error - non-DRAM width sharded input B requires input A to be interleaved or height "
                            "sharded, rather than {}",
                            tensor_a_memory_layout);
                        TT_FATAL(
                            program_config.per_core_N ==
                                (input_tensor_b.shard_spec().value().shape[1] / in1_tile_shape[1]),
                            "Error");
                    }
                    TT_FATAL(
                        input_tensor_b.shard_spec()->grid.bounding_box().start_coord.y ==
                            input_tensor_b.shard_spec()->grid.bounding_box().end_coord.y,
                        "Error");
                }

                if (this->output_mem_config.is_sharded()) {
                    TT_FATAL(this->output_mem_config.memory_layout() == TensorMemoryLayout::BLOCK_SHARDED, "Error");
                    uint32_t M = input_tensor_a.volume() / input_tensor_a.get_padded_shape()[-1] / in0_tile_shape[0];
                    uint32_t N = input_tensor_b.get_padded_shape()[-1] / in1_tile_shape[1];
                    uint32_t per_core_M = program_config.per_core_M;
                    uint32_t per_core_N = program_config.per_core_N;

                    TT_FATAL(
                        program_config.out_subblock_w == per_core_N || program_config.out_subblock_h == 1,
                        "Error: out_subblock_w must be equal to per_core_N or out_subblock_h must be equal to 1.");
                    TT_FATAL(
                        program_config.out_block_w == per_core_N || program_config.out_block_h == 1,
                        "Error: out_block_w must be equal to per_core_N or out_block_h must be equal to 1.");
                }
            } else if constexpr (std::is_same_v<ProgramConfigType, MatmulMultiCoreReuseProgramConfig>) {
                uint32_t M = input_tensor_a.get_padded_shape()[-2] / in0_tile_shape[0];
                uint32_t total_M = input_tensor_a.volume() / input_tensor_a.get_padded_shape()[-1] / in0_tile_shape[0];
                uint32_t N = input_tensor_b.get_padded_shape()[-1] / in1_tile_shape[1];
                uint32_t K = input_tensor_a.get_padded_shape()[-1];
                uint32_t per_core_M = program_config.per_core_M;
                uint32_t per_core_N = program_config.per_core_N;
                if (per_core_M > M) {
                    TT_FATAL(
                        per_core_M % M == 0,
                        "per_core_M, {}, must be a multiple of M, {} if "
                        "per_core_M > M!",
                        per_core_M,
                        M);
                    TT_FATAL(
                        total_M % per_core_M == 0,
                        "input a total height, {}, must be divisible by "
                        "per_core_M, {}!",
                        total_M,
                        per_core_M);
                } else {
                    TT_FATAL(
                        M % per_core_M == 0, "per_core_M, {}, must divide M, {}, if per_core_M < M!", per_core_M, M);
                }
                TT_FATAL(N == per_core_N, "Error: N, {}, is not equal to per_core_N, {}", N, per_core_N);
                if (input_tensor_a.is_sharded()) {
                    TT_FATAL(
                        input_tensor_a.memory_config().memory_layout() != TensorMemoryLayout::WIDTH_SHARDED,
                        "Error: memory layout, {}, is not width sharded",
                        input_tensor_a.memory_config().memory_layout());
                    auto in0_shard_shape = input_tensor_a.shard_spec().value().shape;

                    TT_FATAL(K == in0_shard_shape[1], "Error: K, {}, needs to be equal to {}", K, in0_shard_shape[1]);
                    TT_FATAL(
                        in0_shard_shape[1] == program_config.in0_block_w * in0_tile_shape[1],
                        "Error: {} needs to equal {} * {}",
                        in0_shard_shape[1],
                        program_config.in0_block_w,
                        in0_tile_shape[1]);
                    TT_FATAL(
                        per_core_M * in0_tile_shape[0] == in0_shard_shape[0],
                        "Error: {} * {} needs to equal {}",
                        per_core_M,
                        in0_tile_shape[0],
                        in0_shard_shape[0]);

                    if (input_tensor_b.is_sharded()) {
                        TT_FATAL(
                            input_tensor_a.memory_config().buffer_type() ==
                                input_tensor_b.memory_config().buffer_type(),
                            "Error");
                        TT_FATAL(
                            input_tensor_a.memory_config().memory_layout() ==
                                input_tensor_b.memory_config().memory_layout(),
                            "Error");
                        TT_FATAL(
                            input_tensor_a.shard_spec().value().grid == input_tensor_b.shard_spec().value().grid,
                            "Error");
                        TT_FATAL(
                            input_tensor_a.shard_spec().value().orientation ==
                                input_tensor_b.shard_spec().value().orientation,
                            "Error");
                    }
                    if (this->output_mem_config.is_sharded()) {
                        TT_FATAL(
                            input_tensor_a.memory_config().buffer_type() == this->output_mem_config.buffer_type(),
                            "Error");
                        TT_FATAL(
                            input_tensor_a.memory_config().memory_layout() == this->output_mem_config.memory_layout(),
                            "Error");
                    }
                }

                uint32_t batch_size_a = get_batch_size(input_tensor_a.get_padded_shape());
                uint32_t batch_size_b = get_batch_size(input_tensor_b.get_padded_shape());
                bool broadcast_batch = batch_size_a > 1 and batch_size_b == 1;
                TT_FATAL(!broadcast_batch, "Error");

                if (input_tensor_b.is_sharded()) {
                    TT_FATAL(per_core_M % M == 0, "per_core_M must be a multiple of M if input b is sharded!");
                    TT_FATAL(
                        input_tensor_b.memory_config().memory_layout() != TensorMemoryLayout::WIDTH_SHARDED, "Error");
                    auto in1_shard_shape = input_tensor_b.shard_spec().value().shape;
                    TT_FATAL(in1_shard_shape[1] == input_tensor_b.get_padded_shape()[-1], "Error");
                    TT_FATAL(per_core_N * in1_tile_shape[1] == in1_shard_shape[1], "Error");
                    TT_FATAL(in1_shard_shape[0] % K == 0, "Error");
                }
                if (this->output_mem_config.is_sharded()) {
                    TT_FATAL(this->output_mem_config.memory_layout() != TensorMemoryLayout::WIDTH_SHARDED, "Error");
                    TT_FATAL(
                        program_config.out_subblock_w == per_core_N || program_config.out_subblock_h == 1, "Error");
                }
            } else {
                TT_FATAL(input_tensor_a.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED, "Error");
                TT_FATAL(input_tensor_b.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED, "Error");
                TT_FATAL(this->output_mem_config.memory_layout() == TensorMemoryLayout::INTERLEAVED, "Error");
            }
            if constexpr (
                std::is_same_v<ProgramConfigType, MatmulMultiCoreReuseProgramConfig> ||
                std::is_same_v<ProgramConfigType, MatmulMultiCoreReuseMultiCastProgramConfig> ||
                std::is_same_v<ProgramConfigType, MatmulMultiCoreReuseMultiCast1DProgramConfig>) {
                TT_FATAL(
                    (input_tensor_a.get_padded_shape()[-1] / in0_tile_shape[1]) % program_config.in0_block_w == 0,
                    "Kt must be divisible by in0_block_w");
                TT_FATAL(
                    program_config.per_core_M % program_config.out_subblock_h == 0,
                    "per_core_M must be divisible by out_subblock_h");
                TT_FATAL(
                    program_config.per_core_N % program_config.out_subblock_w == 0,
                    "per_core_N must be divisible by out_subblock_w");
                uint32_t available_reg_count = ttnn::get_dest_reg_count(
                    this->compute_kernel_config.value(), this->output_tile.value().get_tile_shape());
                TT_FATAL(
                    (program_config.out_subblock_w * program_config.out_subblock_h) <= available_reg_count,
                    "out_subblock_w {} times out_subblock_h {} needs to be at "
                    "most {} to fit in hardware",
                    program_config.out_subblock_w,
                    program_config.out_subblock_h,
                    available_reg_count);
            }
        },
        chosen_program_config);
}

std::vector<ttnn::TensorSpec> Matmul::compute_output_specs(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<Tensor>>& optional_output_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors) const {
    TT_FATAL(
        optional_output_tensors.size() <= 1,
        "None or One Optional output tensor can be passed when accessing it "
        "for computing Matmul's output specs");

    const bool is_optional_output_tensor =
        !optional_output_tensors.empty() && optional_output_tensors.at(0).has_value();

    if (is_optional_output_tensor) {
        return {optional_output_tensors.at(0)->get_tensor_spec()};
    }

    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);

    // Use the compute_matmul_output_shape function to get the output shape
    const auto output_shape = compute_matmul_output_shape(input_tensor_a, input_tensor_b);

    auto in0_tile_shape = input_tensor_a.get_tensor_spec().tile().get_tile_shape();
    auto in1_tile_shape = input_tensor_b.get_tensor_spec().tile().get_tile_shape();
    auto output_tile = this->output_tile.value();
    auto tile_width_ratio = output_tile.get_tile_shape()[1] / in1_tile_shape[1];
    auto output_layout = this->untilize_out ? Layout::ROW_MAJOR : Layout::TILE;

    TT_FATAL(this->output_dtype.has_value(), "Error: output_dtype field should have been populated");
    if (this->output_mem_config.is_sharded()) {
        const auto& optional_bias = optional_input_tensors.at(0);
        uint32_t bias_single_tile_size = 0;
        if (optional_bias.has_value()) {
            auto bias_data_format = tt_metal::datatype_to_dataformat_converter(optional_bias.value().get_dtype());
            bias_single_tile_size = tt_metal::detail::TileSize(bias_data_format);
        }
        MatmulProgramConfig chosen_program_config =
            get_program_config(input_tensor_a, input_tensor_b, bias_single_tile_size, this);
        return std::visit(
            [&](const auto& program_config) -> std::vector<TensorSpec> {
                using ProgramConfigType = std::decay_t<decltype(program_config)>;
                if constexpr (std::is_same_v<ProgramConfigType, MatmulMultiCoreReuseMultiCast1DProgramConfig>) {
                    uint32_t M =
                        (program_config.fuse_batch ? input_tensor_a.volume() / input_tensor_a.get_padded_shape()[-1]
                                                   : input_tensor_a.get_padded_shape()[-2]) /
                        in0_tile_shape[0];
                    uint32_t N = input_tensor_b.get_padded_shape()[-1] / in1_tile_shape[1];
                    uint32_t per_core_M = program_config.per_core_M;
                    uint32_t per_core_N = program_config.per_core_N;

                    TT_FATAL(
                        per_core_N % tile_width_ratio == 0,
                        "per_core_N must be divisible by override output tile width");
                    auto mem_config = this->output_mem_config;
                    if (!program_config.gather_in0) {
                        uint32_t num_blocks_y = (M - 1) / per_core_M + 1;
                        uint32_t num_blocks_x = (N - 1) / per_core_N + 1;
                        uint32_t num_blocks_total = num_blocks_y * num_blocks_x;
                        uint32_t num_cores = num_blocks_x * num_blocks_y;
                        CoreRangeSet all_cores =
                            num_cores_to_corerangeset(num_cores, program_config.compute_with_storage_grid_size, true);
                        ShardSpec shard_spec = ShardSpec{
                            all_cores,
                            {per_core_M * in0_tile_shape[0], per_core_N * in1_tile_shape[1]},
                            ShardOrientation::ROW_MAJOR};
                        mem_config = mem_config.with_shard_spec(shard_spec);
                    }
                    // support for multi-tensor output
                    const ttnn::TensorSpec tensor_spec(
                        output_shape,
                        TensorLayout(output_dtype.value(), PageConfig(output_layout, output_tile), mem_config));
                    std::vector<ttnn::TensorSpec> output_tensor_specs(input_tensors.size() - 1, tensor_spec);
                    return output_tensor_specs;
                } else if constexpr (std::is_same_v<
                                         ProgramConfigType,
                                         MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig>) {
                    uint32_t M = input_tensor_a.volume() / input_tensor_a.get_padded_shape()[-1] / in0_tile_shape[0];
                    uint32_t N = input_tensor_b.get_padded_shape()[-1] / in1_tile_shape[1];

                    uint32_t per_core_M = program_config.per_core_M;
                    uint32_t per_core_N = program_config.per_core_N;

                    TT_FATAL(
                        per_core_N % tile_width_ratio == 0,
                        "per_core_N must be divisible by override output tile width");

                    uint32_t num_blocks_y = (M - 1) / per_core_M + 1;
                    uint32_t num_blocks_x = (N - 1) / per_core_N + 1;
                    uint32_t num_cores = num_blocks_x * num_blocks_y;
                    auto end_core = input_tensor_a.shard_spec()->grid.bounding_box().end_coord;
                    auto grid_size = CoreCoord{end_core.x + 1, end_core.y + 1};
                    CoreRangeSet all_cores = num_cores_to_corerangeset(num_cores, grid_size, true);
                    ShardSpec shard_spec = ShardSpec{
                        all_cores,
                        {per_core_M * in0_tile_shape[0], per_core_N * in1_tile_shape[1]},
                        ShardOrientation::ROW_MAJOR};
                    auto mem_config = this->output_mem_config.with_shard_spec(shard_spec);
                    return {TensorSpec(
                        output_shape,
                        TensorLayout(output_dtype.value(), PageConfig(output_layout, output_tile), mem_config))};
                } else if constexpr (std::is_same_v<ProgramConfigType, MatmulMultiCoreReuseMultiCastProgramConfig>) {
                    uint32_t M = input_tensor_a.volume() / input_tensor_a.get_padded_shape()[-1] / in0_tile_shape[0];
                    uint32_t N = input_tensor_b.get_padded_shape()[-1] / in1_tile_shape[1];
                    uint32_t per_core_M = program_config.per_core_M;
                    uint32_t per_core_N = program_config.per_core_N;

                    TT_FATAL(
                        per_core_N % tile_width_ratio == 0,
                        "per_core_N must be divisible by override output tile width");

                    uint32_t num_blocks_y = (M - 1) / per_core_M + 1;
                    uint32_t num_blocks_x = (N - 1) / per_core_N + 1;
                    CoreRangeSet all_cores;
                    ShardOrientation shard_orientation;
                    if (program_config.transpose_mcast) {
                        all_cores = CoreRangeSet({CoreRange({0, 0}, {num_blocks_y - 1, num_blocks_x - 1})});
                        shard_orientation = ShardOrientation::COL_MAJOR;
                    } else {
                        all_cores = CoreRangeSet({CoreRange({0, 0}, {num_blocks_x - 1, num_blocks_y - 1})});
                        shard_orientation = ShardOrientation::ROW_MAJOR;
                    }
                    ShardSpec shard_spec = ShardSpec{
                        all_cores, {per_core_M * in0_tile_shape[0], per_core_N * in1_tile_shape[1]}, shard_orientation};
                    auto mem_config = this->output_mem_config.with_shard_spec(shard_spec);
                    return {TensorSpec(
                        output_shape,
                        TensorLayout(output_dtype.value(), PageConfig(output_layout, output_tile), mem_config))};
                } else if constexpr (std::is_same_v<ProgramConfigType, MatmulMultiCoreReuseProgramConfig>) {
                    uint32_t M = input_tensor_a.volume() / input_tensor_a.get_padded_shape()[-1] / in0_tile_shape[0];
                    uint32_t N = input_tensor_b.get_padded_shape()[-1] / in1_tile_shape[1];
                    uint32_t per_core_M = program_config.per_core_M;
                    uint32_t per_core_N = program_config.per_core_N;

                    TT_FATAL(
                        per_core_N % tile_width_ratio == 0,
                        "per_core_N must be divisible by override output tile width");

                    uint32_t num_blocks_y = (M - 1) / per_core_M + 1;
                    uint32_t num_blocks_x = (N - 1) / per_core_N + 1;
                    uint32_t num_cores = num_blocks_x * num_blocks_y;
                    ShardOrientation shard_orientation = ShardOrientation::COL_MAJOR;
                    if (input_tensor_a.is_sharded()) {
                        shard_orientation = input_tensor_a.shard_spec().value().orientation;
                    } else if (input_tensor_b.is_sharded()) {
                        shard_orientation = input_tensor_b.shard_spec().value().orientation;
                    }

                    CoreRangeSet all_cores = num_cores_to_corerangeset(
                        num_cores,
                        program_config.compute_with_storage_grid_size,
                        shard_orientation == ShardOrientation::ROW_MAJOR);
                    ShardSpec shard_spec = ShardSpec{
                        all_cores, {per_core_M * in0_tile_shape[0], per_core_N * in1_tile_shape[1]}, shard_orientation};
                    auto mem_config = this->output_mem_config.with_shard_spec(shard_spec);
                    return {TensorSpec(
                        output_shape,
                        TensorLayout(output_dtype.value(), PageConfig(output_layout, output_tile), mem_config))};
                } else {
                    TT_FATAL(
                        in0_tile_shape[0] == TILE_HEIGHT and in0_tile_shape[1] == TILE_WIDTH,
                        "matmul with non-optimized program config does not "
                        "support tiny tile");
                    TT_FATAL(
                        in1_tile_shape[0] == TILE_HEIGHT and in1_tile_shape[1] == TILE_WIDTH,
                        "matmul with non-optimized program config does not "
                        "support tiny tile");
                    if (this->output_tile.has_value()) {
                        TT_FATAL(
                            this->output_tile->get_tile_shape()[0] == TILE_HEIGHT and
                                this->output_tile->get_tile_shape()[1] == TILE_WIDTH,
                            "matmul with non-optimized program config does not "
                            "support tiny tile");
                    }
                    TT_THROW("Unsupported op for output sharding");
                    return {};
                }
            },
            chosen_program_config);
    }

    return {TensorSpec(
        output_shape, TensorLayout(output_dtype.value(), PageConfig(Layout::TILE, output_tile), output_mem_config))};
}

std::vector<Tensor> Matmul::create_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& optional_output_tensors) const {
    return operation::default_create_output_tensors(*this, input_tensors, optional_output_tensors);
}

using MatmulCallback = tt::tt_metal::operation::OverrideRuntimeArgumentsCallback<std::vector<Tensor>>;

MeshCoordinateRange get_range_from_mesh_coords(const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    TT_FATAL(tensor_coords.size() == 1, "Cannot support dispatching TTNN Ops to different device ranges.");
    return tensor_coords.ranges().front();
}

operation::CacheableMeshWorkload<std::vector<Tensor>> create_homogenous_mesh_workload(
    tt::tt_metal::operation::ProgramWithCallbacks& matmul_program, const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    tt::tt_metal::distributed::MeshWorkload matmul_workload = tt::tt_metal::distributed::CreateMeshWorkload();
    std::unordered_map<MeshCoordinateRange, MatmulCallback> callbacks = {};

    auto workload_device_range = get_range_from_mesh_coords(tensor_coords);
    AddProgramToMeshWorkload(matmul_workload, std::move(matmul_program.program), workload_device_range);
    callbacks[workload_device_range] = std::move(matmul_program.override_runtime_arguments_callback.value());
    return {.workload = std::move(matmul_workload), .per_program_callbacks = std::move(callbacks)};
}

operation::CacheableMeshWorkload<std::vector<Tensor>> Matmul::create_mesh_workload(
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);
    const auto& bias = optional_input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);

    TT_FATAL(this->output_dtype.has_value(), "Error: output_dtype field should have been populated");
    tt::tt_metal::DataType output_dtype = this->output_dtype.value();

    bool fuse_batch = true;
    // TODO: If input_tensor_a.get_padded_shape()[0] * input_tensor_a.get_padded_shape()[1] * ... except last two
    // dimensions == 1, does matmuls work if we treat it as bmm
    // TODO: Only for MatmulMultiCoreReuseProgramConfig we allow this as single core matmul/bmm
    TT_FATAL(this->compute_kernel_config.has_value(), "Error: compute_kernel_config field should have been populated");
    TT_FATAL(this->bcast_batch.has_value(), "Error: bcast_batch field should have been populated");
    bool broadcast_batch = this->bcast_batch.value();
    uint32_t bias_single_tile_size = 0;
    if (bias.has_value()) {
        auto bias_data_format = tt_metal::datatype_to_dataformat_converter(bias.value().get_dtype());
        bias_single_tile_size = tt_metal::detail::TileSize(bias_data_format);
    }

    MatmulProgramConfig chosen_program_config =
        get_program_config(input_tensor_a, input_tensor_b, bias_single_tile_size, this);

    auto mesh_device = input_tensor_a.mesh_device();

    return std::visit(
        [&](const auto& program_config) -> tt::tt_metal::operation::CacheableMeshWorkload<std::vector<Tensor>> {
            using ProgramConfigType = std::decay_t<decltype(program_config)>;
            if constexpr (std::is_same_v<ProgramConfigType, MatmulMultiCoreReuseProgramConfig>) {
                TT_FATAL(!bias.has_value(), "Bias is not supported for MatmulMultiCoreReuseProgramConfig!");
                // TODO: fuse_batch doesn't do anything for this variant! Code is
                // doing fuse_batch=false
                auto bmm_program = bmm_multi_core_reuse_optimized(
                    input_tensor_a,
                    input_tensor_b,
                    output_tensor,
                    broadcast_batch,
                    program_config.compute_with_storage_grid_size,
                    output_dtype,
                    this->compute_kernel_config.value(),
                    program_config.in0_block_w,
                    program_config.out_subblock_h,
                    program_config.out_subblock_w,
                    program_config.per_core_M,
                    program_config.per_core_N,
                    /*fuse_batch=*/false,
                    this->untilize_out);

                return create_homogenous_mesh_workload(bmm_program, tensor_coords);

            } else if constexpr (std::is_same_v<ProgramConfigType, MatmulMultiCoreReuseMultiCastProgramConfig>) {
                auto mcast_mm_program = matmul_multi_core_reuse_mcast_2d_optimized(
                    input_tensor_a,
                    input_tensor_b,
                    bias,
                    output_tensor,
                    broadcast_batch,
                    program_config.compute_with_storage_grid_size,
                    this->compute_kernel_config.value(),
                    program_config.in0_block_w,
                    program_config.out_subblock_h,
                    program_config.out_subblock_w,
                    program_config.out_block_h,
                    program_config.out_block_w,
                    program_config.per_core_M,
                    program_config.per_core_N,
                    program_config.fuse_batch,
                    program_config.transpose_mcast,
                    program_config.fused_activation,
                    this->untilize_out);

                return create_homogenous_mesh_workload(mcast_mm_program, tensor_coords);

            } else if constexpr (std::is_same_v<ProgramConfigType, MatmulMultiCoreReuseMultiCast1DProgramConfig>) {
                const std::vector<Tensor> input_tensors_b(input_tensors.begin() + 1, input_tensors.end());
                auto mcast_mm_program = matmul_multi_core_reuse_mcast_1d_optimized(
                    input_tensor_a,
                    input_tensors_b,
                    bias,
                    output_tensors,
                    broadcast_batch,
                    program_config.compute_with_storage_grid_size,
                    this->compute_kernel_config.value(),
                    program_config.in0_block_w,
                    program_config.out_subblock_h,
                    program_config.out_subblock_w,
                    program_config.out_block_h,
                    program_config.out_block_w,
                    program_config.per_core_M,
                    program_config.per_core_N,
                    program_config.fuse_batch,
                    program_config.fused_activation,
                    program_config.mcast_in0,
                    program_config.gather_in0,
                    program_config.hop_cores,
                    this->untilize_out,
                    this->global_cb,
                    program_config.num_global_cb_receivers,
                    this->sub_device_id);

                return create_homogenous_mesh_workload(mcast_mm_program, tensor_coords);

            } else if constexpr (std::is_same_v<
                                     ProgramConfigType,
                                     MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig>) {
                // DRAM Sharded Matmul generates different programs across devices, since it depends on harvesting.
                // Account for this by creating a heterogenous MeshWorkload.
                auto workload_device_range = get_range_from_mesh_coords(tensor_coords);
                tt::tt_metal::distributed::MeshWorkload dram_sharded_mm_workload;
                std::unordered_map<MeshCoordinateRange, MatmulCallback> callbacks;

                for (const auto& coord : workload_device_range) {
                    auto dram_sharded_mm_program = matmul_multi_core_reuse_dram_sharded_optimized(
                        coord,
                        input_tensor_a,
                        input_tensor_b,
                        bias,
                        output_tensor,
                        this->compute_kernel_config.value(),
                        program_config.in0_block_w,
                        program_config.per_core_M,
                        program_config.per_core_N,
                        program_config.fused_activation,
                        this->untilize_out,
                        false,
                        false,
                        false);
                    AddProgramToMeshWorkload(
                        dram_sharded_mm_workload,
                        std::move(dram_sharded_mm_program.program),
                        MeshCoordinateRange(coord, coord));
                    callbacks[MeshCoordinateRange(coord, coord)] =
                        std::move(dram_sharded_mm_program.override_runtime_arguments_callback.value());
                }
                return {.workload = std::move(dram_sharded_mm_workload), .per_program_callbacks = std::move(callbacks)};

            } else if constexpr (std::is_same_v<ProgramConfigType, MatmulMultiCoreProgramConfig>) {
                TT_FATAL(!bias.has_value(), "Bias is not supported for matmul multi core");
                auto multicore_mm_program =
                    matmul_multi_core(input_tensor_a, input_tensor_b, output_tensor, broadcast_batch);
                return create_homogenous_mesh_workload(multicore_mm_program, tensor_coords);
            } else {
                TT_THROW("Unrecognized Config");
            }
        },
        chosen_program_config);
}

operation::OpPerformanceModel Matmul::create_op_performance_model(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    std::vector<Tensor>& output_tensors) const {
    return ::create_op_performance_model_for_matmul(
        input_tensors, optional_input_tensors, output_tensors, this->compute_kernel_config.value());
}

}  // namespace matmul

}  // namespace operations

}  // namespace ttnn

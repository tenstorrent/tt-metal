// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/matmul/device/matmul_op.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <optional>

#include "tt_metal/common/constants.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/hostdevcommon/common_values.hpp"
#include "tt_metal/common/work_split.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/types.hpp"

using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal;
using tt::tt_metal::LegacyShape;
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

bool get_broadcast_batch(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const std::optional<const MatmulProgramConfig> matmul_program_config) {
    uint32_t batch_size_b = get_batch_size(input_tensor_b.get_legacy_shape());
    bool broadcast_batch = batch_size_b == 1;
    if (!matmul_program_config.has_value()) {
        return broadcast_batch;
    }

    bool is_multi_core_reuse = std::visit(
        [](const auto& program_config) -> bool {
            using ProgramConfigType = std::decay_t<decltype(program_config)>;
            if constexpr (std::is_same_v<ProgramConfigType, MatmulMultiCoreReuseProgramConfig>) {
                return true;
            }
            return false;
        },
        matmul_program_config.value());
    if (is_multi_core_reuse) {
        uint32_t batch_size_a = get_batch_size(input_tensor_a.get_legacy_shape());
        broadcast_batch &= batch_size_a > 1;
    }
    return broadcast_batch;
}

operation::OpPerformanceModel create_op_performance_model_for_matmul(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    const std::vector<Tensor>& output_tensors,
    const ttnn::DeviceComputeKernelConfig& compute_kernel_config) {
    const auto& in_a_shape = input_tensors.at(0).get_shape();
    const auto& in_b_shape = input_tensors.at(1).get_shape();
    const auto& out_shape = output_tensors.at(0).get_shape();

    const auto& t = output_tensors.at(0);
    if (t.storage_type() != StorageType::DEVICE) {
        tt::log_warning(tt::LogOp, "Output tensor not on DEVICE?!");
    }

    auto arch = t.storage_type() == StorageType::DEVICE ? t.device()->arch() : ttnn::operations::experimental::auto_format::AutoFormat::GetDefaultDevice()->arch();
    const int num_cores = (arch == ARCH::WORMHOLE_B0) ? 8 * 8 : 9 * 12;
    const int tensix_mul_adds_per_cycle_lofi = (arch == ARCH::WORMHOLE_B0) ? 4096 : 2048;

    // Calculate number of mul/add operations
    // TODO: add bias modeling
    int64_t num_mul_adds_per_elem = in_a_shape[-1] * 2;  // 1 multiply and 1 add per element
    uint32_t batch_size = get_batch_size(out_shape);
    int64_t num_mul_adds = num_mul_adds_per_elem * out_shape[-2] * out_shape[-1] * batch_size;

    MathFidelity math_fidelity = MathFidelity::Invalid;

    std::visit(
        [&](auto&& compute_kernel_config) {
            using T = std::decay_t<decltype(compute_kernel_config)>;
            if constexpr (std::is_same_v<T, ttnn::GrayskullComputeKernelConfig>) {
                math_fidelity = compute_kernel_config.math_fidelity;
            } else if constexpr (std::is_same_v<T, ttnn::WormholeComputeKernelConfig>) {
                math_fidelity = compute_kernel_config.math_fidelity;
            } else {
                TT_THROW("arch not supported");
            }
        },
        compute_kernel_config);

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
    uint32_t in0_single_tile_size,
    uint32_t in1_single_tile_size,
    uint32_t output_single_tile_size) {
    // Circular Buffer sizes:
    // src0 CB: per_core_M * in0_block_w * 2 (for double buffer)
    // src1 CB: per_core_N * in0_block_w * 2 (for double buffer)
    // out CB:  per_core_M * per_core_N
    // Ignore optional intermediate CB because not needed when need to create a program config.
    uint32_t in0_size = per_core_M * in0_block_w * 2 * in0_single_tile_size;
    uint32_t in1_size = per_core_M * in0_block_w * 2 * in1_single_tile_size;
    uint32_t out_size = per_core_M * per_core_N * output_single_tile_size;
    return in0_size + in1_size + out_size;
}

inline uint32_t get_max_l1_space(const Tensor& input_tensor_a) {
    tt::tt_metal::Device* device = input_tensor_a.device();
    const std::vector<uint32_t>& bank_ids =
        device->bank_ids_from_logical_core(BufferType::L1, *device->compute_cores_.begin());
    std::optional<uint64_t> lowest_address = allocator::lowest_occupied_l1_address(*device->allocator_, bank_ids[0]);
    uint32_t max_l1_space = lowest_address.has_value() ? lowest_address.value() : device->l1_size_per_core();
    max_l1_space = max_l1_space - L1_UNRESERVED_BASE;
    return max_l1_space;
}

inline bool can_cbs_fit_in_l1(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    uint32_t per_core_M,
    uint32_t per_core_N,
    uint32_t in0_block_w) {
    uint32_t max_l1_space = get_max_l1_space(input_tensor_a);
    tt::DataFormat in0_data_format = tt_metal::datatype_to_dataformat_converter(input_tensor_a.get_dtype());
    tt::DataFormat in1_data_format = tt_metal::datatype_to_dataformat_converter(input_tensor_b.get_dtype());
    uint32_t in0_single_tile_size = tt_metal::detail::TileSize(in0_data_format);  // use as estimate for output as well
    uint32_t in1_single_tile_size = tt_metal::detail::TileSize(in1_data_format);
    uint32_t size = get_estimated_size_of_cbs(
        per_core_M, per_core_N, in0_block_w, in0_single_tile_size, in1_single_tile_size, in0_single_tile_size);
    return size < max_l1_space;
}

inline uint32_t get_per_core_factor(const Tensor& input_tensor_a, const Tensor& input_tensor_b, uint32_t in0_block_w) {
    uint32_t max_l1_space = get_max_l1_space(input_tensor_a);
    tt::DataFormat in0_data_format = tt_metal::datatype_to_dataformat_converter(input_tensor_a.get_dtype());
    tt::DataFormat in1_data_format = tt_metal::datatype_to_dataformat_converter(input_tensor_b.get_dtype());
    uint32_t in0_single_tile_size = tt_metal::detail::TileSize(in0_data_format);  // use as estimate for output as well
    uint32_t in1_single_tile_size = tt_metal::detail::TileSize(in1_data_format);
    for (uint32_t per_core_factor = 16; per_core_factor > 1; per_core_factor /= 2) {
        uint32_t size = get_estimated_size_of_cbs(
            per_core_factor,
            per_core_factor,
            in0_block_w,
            in0_single_tile_size,
            in1_single_tile_size,
            in0_single_tile_size);
        if (size < max_l1_space) {
            return per_core_factor;
        }
    }
    return 1;
}

MatmulProgramConfig create_matmul_1d_systolic_array_program_config(
    const ttnn::types::Shape& input_shape_a,
    const ttnn::types::Shape& input_shape_b,
    const CoreCoord& core_coord,
    const std::optional<const UnaryWithParam> fused_activation,
    const bool fp32_dest_acc_en,
    const TensorMemoryLayout input_layout_a) {
    auto a_padded_shape = input_shape_a.with_tile_padding();
    auto b_padded_shape = input_shape_b.with_tile_padding();
    auto k_size = a_padded_shape[-1];
    auto m_size = a_padded_shape[-2];
    auto n_size = b_padded_shape[-1];
    uint32_t batch_size_a = get_batch_size(a_padded_shape);
    uint32_t batch_size_b = get_batch_size(b_padded_shape);
    bool input_b_is_batched = batch_size_b > 1;
    TT_FATAL(batch_size_b == 1, "Second input cannot be currently batched when running matmul using 1d systolic array");
    TT_FATAL(
        (batch_size_a * m_size) % ttnn::TILE_SIZE == 0 && k_size % ttnn::TILE_SIZE == 0 &&
            n_size % ttnn::TILE_SIZE == 0,
        "The last two dimensions of the first tensor and the last dimension of the second tensor must be a multiple of "
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
    auto matmul_params = get_subblock_sizes(batch_and_m_tiles_per_core, n_tiles_per_core, fp32_dest_acc_en);
    uint32_t out_subblock_h = std::get<0>(matmul_params);
    uint32_t out_subblock_w = std::get<1>(matmul_params);
    return MatmulMultiCoreReuseMultiCast1DProgramConfig{
        .compute_with_storage_grid_size = {core_coord.x, core_coord.y},
        .in0_block_w = k_tiles_per_core,
        .out_subblock_h = out_subblock_h,
        .out_subblock_w = out_subblock_w,
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
    const bool fuse_batch,
    const std::optional<UnaryWithParam> fused_activation,
    const bool mcast_in0,
    const bool out_sharded,
    const std::optional<const CoreCoord> compute_with_storage_grid_size,
    const std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config) {
    auto device = input_tensor_a.device();
    auto grid_size = compute_with_storage_grid_size.value_or(device->compute_with_storage_grid_size());
    uint32_t M = fuse_batch ? input_tensor_a.volume() / input_tensor_a.get_legacy_shape()[-1]
                            : input_tensor_a.get_legacy_shape()[-2];
    uint32_t K = input_tensor_a.get_legacy_shape()[-1];
    uint32_t N = input_tensor_b.get_legacy_shape()[-1];
    uint32_t per_core_M, per_core_N;
    auto in0_tile_shape = input_tensor_a.get_tile().get_tile_shape();
    auto in1_tile_shape = input_tensor_b.get_tile().get_tile_shape();
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
    auto subblock_hw = bmm_op_utils::get_matmul_subblock_params(
        per_core_M,
        per_core_N,
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
        .per_core_M = per_core_M,
        .per_core_N = per_core_N,
        .fuse_batch = fuse_batch,
        .fused_activation = fused_activation,
        .mcast_in0 = mcast_in0};
}

inline MatmulProgramConfig create_simple_matmul_program_config(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config,
    const CoreCoord& compute_with_storage_grid_size) {
    const auto &ashape = input_tensor_a.get_legacy_shape(), bshape = input_tensor_b.get_legacy_shape();
    uint32_t batch_size_a = get_batch_size(ashape);
    uint32_t num_output_tiles = batch_size_a * ashape[-2] * bshape[-1] / TILE_HW;  // Output M x N

    auto in0_tile_shape = input_tensor_a.get_tile().get_tile_shape();
    auto in1_tile_shape = input_tensor_b.get_tile().get_tile_shape();

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

    // out_subblock h/w doesn't matter
    per_core_M = get_per_core_factor(input_tensor_a, input_tensor_b, in0_block_w);
    per_core_N = per_core_M;

    // Calculate number of blocks along x and y; tensor dims are padded up to 512
    num_blocks_y = (Mt - 1) / per_core_M + 1;
    num_blocks_x = (Nt - 1) / per_core_N + 1;

    if (num_blocks_x * num_blocks_y <= num_cores_x * num_cores_y and Kt % in0_block_w == 0) {
        CoreCoord core_range = get_core_range(num_blocks_y, num_blocks_x, num_cores_y, num_cores_x);
        if (core_range.y == 1) {
            return get_mcast_1d_config(
                input_tensor_a,
                input_tensor_b,
                false /* fuse_batch */,
                std::nullopt /* fused_activation */,
                true /* mcast_in0 */,
                false /* out_sharded */,
                std::nullopt /* compute_with_storage_grid_size */,
                compute_kernel_config);
        } else if (core_range.x == 1) {
            return get_mcast_1d_config(
                input_tensor_a,
                input_tensor_b,
                false /* fuse_batch */,
                std::nullopt /* fused_activation */,
                false /* mcast_in0 */,
                false /* out_sharded */,
                std::nullopt /* compute_with_storage_grid_size */,
                compute_kernel_config);
        } else if (core_range.y > 0 && num_blocks_x <= num_cores_x && num_blocks_y <= num_cores_y) {
            bool transpose_mcast = input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::BLOCK_SHARDED &&
                                   input_tensor_a.shard_spec().value().orientation == ShardOrientation::COL_MAJOR;
            out_subblock_h = 4;
            out_subblock_w = 2;
            if (out_subblock_w != per_core_N) {
                out_subblock_h = 1;
            }
            return MatmulMultiCoreReuseMultiCastProgramConfig{
                .compute_with_storage_grid_size = {num_cores_x, num_cores_y},
                .in0_block_w = in0_block_w,
                .out_subblock_h = out_subblock_h,
                .out_subblock_w = out_subblock_w,
                .per_core_M = per_core_M,
                .per_core_N = per_core_N,
                .transpose_mcast = transpose_mcast,
                .fused_activation = std::nullopt,
                .fuse_batch = false,
            };
        }
        // If we don't need padding, use the default multi_core reuse/reuse_mcast
        else if (Mt % per_core_M == 0 and Nt % per_core_N == 0) {
            return MatmulMultiCoreNonOptimizedReuseProgramConfig{};
        } else {
            return MatmulMultiCoreProgramConfig{};
        }
    } else {
        return MatmulMultiCoreProgramConfig{};
    }
}

MatmulProgramConfig create_matmul_program_config(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const std::optional<const CoreCoord> user_core_coord,
    const std::optional<UnaryWithParam> fused_activation,
    const std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config) {
    auto a_shape = input_tensor_a.get_shape();
    auto b_shape = input_tensor_b.get_shape();
    auto a_padded_shape = a_shape.with_tile_padding();
    auto b_padded_shape = b_shape.with_tile_padding();
    auto a_layout = input_tensor_a.memory_config().memory_layout;
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
        "The last two dimensions of the first tensor and the last dimension of the second tensor must be a multiple of "
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
            k_tiles_per_core = 1;  // TODO(arakhmati): Can it be more than 1 without running out of memory?
            if (!can_cbs_fit_in_l1(
                    input_tensor_a, input_tensor_b, m_tiles_per_core, n_tiles_per_core, k_tiles_per_core)) {
                return create_simple_matmul_program_config(
                    input_tensor_a, input_tensor_b, compute_kernel_config, core_coord);
            }
        } else if (a_is_sharded) {
            TT_FATAL(
                a_layout != TensorMemoryLayout::WIDTH_SHARDED,
                "MatmulMultiCoreReuseProgramConfig: Cannot be width sharded");
            auto shard_shape = input_tensor_a_memory_config.shard_spec.value().shape;
            uint32_t n = b_shape[-1] / ttnn::TILE_SIZE;
            m_tiles_per_core = shard_shape[0] / ttnn::TILE_SIZE;
            n_tiles_per_core = n;
            k_tiles_per_core = shard_shape[1] / ttnn::TILE_SIZE;
        } else {
            TT_FATAL(
                input_tensor_b_memory_config.memory_layout != TensorMemoryLayout::WIDTH_SHARDED,
                "MatmulMultiCoreReuseProgramConfig: Cannot be width sharded");
            auto shard_shape = input_tensor_b_memory_config.shard_spec.value().shape;
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
    if (height_width_ratio > 8 || any_size_within_tile) {
        if (!a_is_block_sharded) {
            return create_matmul_1d_systolic_array_program_config(
                a_shape, b_shape, core_coord, fused_activation, fp32_dest_acc_en, a_layout);
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
                a_shape, b_shape, core_coord, fused_activation, fp32_dest_acc_en, a_layout);
        }
        uint32_t k = a_shape[-1] / ttnn::TILE_SIZE;
        uint32_t n = b_shape[-1] / ttnn::TILE_SIZE;
        auto shard_shape = input_tensor_a_memory_config.shard_spec.value().shape;
        m_tiles_per_core = shard_shape[0] / ttnn::TILE_SIZE;
        n_tiles_per_core = (n * shard_shape[1]) / (k * ttnn::TILE_SIZE);
        k_tiles_per_core = std::gcd(shard_shape[1] / ttnn::TILE_SIZE, k);
    }

    n_tiles_per_core = std::max(n_tiles_per_core, (unsigned int)1);
    auto matmul_params = get_subblock_sizes(m_tiles_per_core, n_tiles_per_core, fp32_dest_acc_en);
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
        .per_core_M = m_tiles_per_core,
        .per_core_N = n_tiles_per_core,
        .transpose_mcast = transpose_mcast,
        .fused_activation = fused_activation,
    };
}

// TODO: Only supports sharded matmul for now; infer most matmul params from shard spec
MatmulProgramConfig get_matmul_program_config(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const MemoryConfig& output_mem_config,
    const std::optional<UnaryWithParam> fused_activation,
    const bool matmul,
    const std::optional<const CoreCoord> user_core_coord,
    const std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config) {
    TT_FATAL(input_tensor_a.is_sharded(), "Error");
    bool fp32_dest_acc_en = get_fp32_dest_acc_en(compute_kernel_config);
    // TODO: allow overwriting of grid size by user_core_coord after allowing support of arbitrary compute grid and more
    // generic sharded output tensor creation
    auto grid_size = input_tensor_a.shard_spec().value().grid.bounding_box().grid_size();

    auto in0_tile_shape = input_tensor_a.get_tile().get_tile_shape();
    auto in1_tile_shape = input_tensor_b.get_tile().get_tile_shape();

    // MCAST matmuls only support input_b in INTERLEAVED
    if (matmul) {
        TT_FATAL(input_tensor_b.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED, "Error");
        if ((input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::WIDTH_SHARDED or
             input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED) and
            (grid_size.x > 1 or grid_size.y > 1)) {
            TT_FATAL(input_tensor_a.shard_spec().value().orientation == ShardOrientation::ROW_MAJOR, "Error");

            bool per_core_N_equals_subblock_w_constraint = false;
            if (output_mem_config.is_sharded()) {
                TT_FATAL(input_tensor_a.memory_config().buffer_type == output_mem_config.buffer_type, "Error");
                TT_FATAL(input_tensor_a.memory_config().memory_layout == output_mem_config.memory_layout, "Error");
                per_core_N_equals_subblock_w_constraint = true;
            }

            uint32_t M = input_tensor_a.volume() / input_tensor_a.get_legacy_shape()[-1] / in0_tile_shape[0];
            uint32_t K = input_tensor_a.get_legacy_shape()[-1] / in0_tile_shape[1];
            uint32_t N = input_tensor_b.get_legacy_shape()[-1] / in1_tile_shape[1];
            auto shard_shape = input_tensor_a.shard_spec().value().shape;

            bool mcast_in0;
            uint32_t per_core_M;
            uint32_t per_core_N;
            uint32_t in0_block_w;
            if (input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::WIDTH_SHARDED) {
                mcast_in0 = true;
                per_core_M = M;
                per_core_N = div_up(N, input_tensor_a.shard_spec().value().grid.num_cores());
                in0_block_w = std::gcd(shard_shape[1] / in0_tile_shape[1], K);
            } else if (input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED) {
                mcast_in0 = false;
                per_core_M = shard_shape[0] / in0_tile_shape[0];
                per_core_N = N;  // Only necessary if output is sharded; otherwise, can set this to be < N
                in0_block_w = K;
            } else {
                TT_THROW("Input tensor must be WIDTH or HEIGHT sharded for 1D mcast matmul!");
            }

            auto subblock_hw = bmm_op_utils::get_matmul_subblock_params(
                per_core_M, per_core_N, false, per_core_N_equals_subblock_w_constraint, fp32_dest_acc_en);
            auto out_subblock_h = std::get<0>(subblock_hw);
            auto out_subblock_w = std::get<1>(subblock_hw);

            return MatmulMultiCoreReuseMultiCast1DProgramConfig{
                .compute_with_storage_grid_size = grid_size,
                .in0_block_w = in0_block_w,
                .out_subblock_h = out_subblock_h,
                .out_subblock_w = out_subblock_w,
                .per_core_M = per_core_M,
                .per_core_N = per_core_N,
                .fuse_batch = true,
                .fused_activation = fused_activation,
                .mcast_in0 = mcast_in0,
            };
        } else if (
            input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::BLOCK_SHARDED and
            (grid_size.x > 1 and grid_size.y > 1)) {
            bool transpose_mcast = input_tensor_a.shard_spec().value().orientation == ShardOrientation::COL_MAJOR;

            bool per_core_N_equals_subblock_w_constraint = false;
            if (output_mem_config.is_sharded()) {
                TT_FATAL(input_tensor_a.memory_config().buffer_type == output_mem_config.buffer_type, "Error");
                TT_FATAL(input_tensor_a.memory_config().memory_layout == output_mem_config.memory_layout, "Error");
                per_core_N_equals_subblock_w_constraint = true;
            }

            uint32_t M = input_tensor_a.volume() / input_tensor_a.get_legacy_shape()[-1] / in0_tile_shape[0];
            uint32_t K = input_tensor_a.get_legacy_shape()[-1] / in0_tile_shape[1];
            uint32_t N = input_tensor_b.get_legacy_shape()[-1] / in1_tile_shape[1];

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

            auto subblock_hw = bmm_op_utils::get_matmul_subblock_params(
                per_core_M, per_core_N, false, per_core_N_equals_subblock_w_constraint, fp32_dest_acc_en);
            auto out_subblock_h = std::get<0>(subblock_hw);
            auto out_subblock_w = std::get<1>(subblock_hw);

            return MatmulMultiCoreReuseMultiCastProgramConfig{
                .compute_with_storage_grid_size = grid_size,
                .in0_block_w = in0_block_w,
                .out_subblock_h = out_subblock_h,
                .out_subblock_w = out_subblock_w,
                .per_core_M = per_core_M,
                .per_core_N = per_core_N,
                .transpose_mcast = transpose_mcast,
                .fused_activation = fused_activation,
            };
        }
    } else {
        // TODO: Need a better criteria for BMMs and MatmulMultiCoreReuseProgramConfig
        TT_FATAL(input_tensor_a.memory_config().memory_layout != TensorMemoryLayout::WIDTH_SHARDED, "Error");

        bool per_core_N_equals_subblock_w_constraint = false;
        if (output_mem_config.is_sharded()) {
            TT_FATAL(input_tensor_a.memory_config().buffer_type == output_mem_config.buffer_type, "Error");
            TT_FATAL(input_tensor_a.memory_config().memory_layout == output_mem_config.memory_layout, "Error");
            per_core_N_equals_subblock_w_constraint = true;
        }

        uint32_t M = input_tensor_a.volume() / input_tensor_a.get_legacy_shape()[-1] / in0_tile_shape[0];
        uint32_t K = input_tensor_a.get_legacy_shape()[-1] / in0_tile_shape[1];
        uint32_t N = input_tensor_b.get_legacy_shape()[-1] / in1_tile_shape[1];

        auto in0_shard_shape = input_tensor_a.shard_spec().value().shape;
        uint32_t per_core_M = in0_shard_shape[0] / in0_tile_shape[0];
        uint32_t per_core_N = N;
        uint32_t in0_block_w = in0_shard_shape[1] / in0_tile_shape[1];

        auto subblock_hw = bmm_op_utils::get_matmul_subblock_params(
            per_core_M, per_core_N, false, per_core_N_equals_subblock_w_constraint, fp32_dest_acc_en);
        auto out_subblock_h = std::get<0>(subblock_hw);
        auto out_subblock_w = std::get<1>(subblock_hw);

        // TODO: Temporarily allow for single core; should support bcast_batch in general
        uint32_t batch_size_a = get_batch_size(input_tensor_a.get_legacy_shape());
        uint32_t batch_size_b = get_batch_size(input_tensor_b.get_legacy_shape());
        bool broadcast_batch = batch_size_a > 1 and batch_size_b == 1;
        TT_FATAL(!broadcast_batch, "Error");

        if (input_tensor_b.is_sharded()) {
            TT_FATAL(input_tensor_a.memory_config().buffer_type == input_tensor_b.memory_config().buffer_type, "Error");
            TT_FATAL(input_tensor_a.memory_config().memory_layout == input_tensor_b.memory_config().memory_layout, "Error");
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
        input_tensor_a, input_tensor_b, user_core_coord, fused_activation, compute_kernel_config);
}

inline MatmulProgramConfig generate_matmul_program_config(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const MemoryConfig& mem_config,
    const std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<const CoreCoord> user_core_coord,
    const std::optional<UnaryWithParam> user_fused_activation,
    const bool user_run_batched) {
    const bool has_user_grid = user_core_coord.has_value();
    if (has_user_grid || !input_tensor_a.is_sharded()) {
        CoreCoord core_coord;
        if (has_user_grid) {
            core_coord = user_core_coord.value();
            return create_matmul_program_config(
                input_tensor_a, input_tensor_b, user_core_coord, user_fused_activation, compute_kernel_config);
        } else {
            tt::tt_metal::Device* device = input_tensor_a.device();
            auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
            return create_simple_matmul_program_config(
                input_tensor_a, input_tensor_b, compute_kernel_config, compute_with_storage_grid_size);
        }
    } else {
        bool bmm = user_run_batched;
        return get_matmul_program_config(
            input_tensor_a, input_tensor_b, mem_config, std::nullopt, !bmm, user_core_coord, compute_kernel_config);
    }
}

inline MatmulProgramConfig get_program_config(
    const Tensor& input_tensor_a, const Tensor& input_tensor_b, const struct Matmul* matmul) {
    if (matmul->program_config.has_value()) {
        return matmul->program_config.value();
    }
    auto config = generate_matmul_program_config(
        input_tensor_a,
        input_tensor_b,
        matmul->output_mem_config,
        matmul->compute_kernel_config,
        matmul->user_core_coord,
        matmul->user_fused_activation,
        matmul->user_run_batched);
    tt::log_debug(tt::LogOp, "Auto generated program config: {}", config);

    // Sanity checks for matmul program configs
    std::visit(
        [input_tensor_a](const auto& program_config) {
            using ProgramConfigType = std::decay_t<decltype(program_config)>;
            if constexpr (
                not std::is_same_v<ProgramConfigType, MatmulMultiCoreProgramConfig> and
                not std::is_same_v<ProgramConfigType, MatmulMultiCoreNonOptimizedReuseProgramConfig> and
                not std::is_same_v<ProgramConfigType, MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig>) {
                TT_FATAL(
                    program_config.compute_with_storage_grid_size.x <=
                        input_tensor_a.device()->compute_with_storage_grid_size().x,
                    "Number of columns in matmul compute grid exceeds maximum device compute grid size!");
                TT_FATAL(
                    program_config.compute_with_storage_grid_size.y <=
                        input_tensor_a.device()->compute_with_storage_grid_size().y,
                    "Number of rows in matmul compute grid exceeds maximum device compute grid size!");
                TT_FATAL(
                    program_config.compute_with_storage_grid_size.x > 0,
                    "Number of columns in matmul compute grid must be greater than 0!");
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
                continue;  // Total number of tiles in a subblock must be less than 4 when in fp32_dest_acc mode
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

void add_stagger_defines_if_needed(
    const tt::ARCH arch, const int num_cores, std::map<string, string>& mm_kernel_defines) {
    // Empirically deduced di/dt problems appear for matmuls using more than 48 cores;
    // when there is 48 cores or less, we never enable stagger since the delay impacts op performance
    constexpr uint32_t WH_B0_MM_MAX_CORES_NO_STAGGER = 48;

    // Apply stagger delay on Wormhole B0 on odd rows, so that only half of cores start doing work at once.
    // This is done to mitigate di/dt issues, in case the environment var is set.
    // See issue #9857.
    const bool enable_stagger = std::getenv("TT_ENABLE_MATMUL_STAGGER");
    if (enable_stagger && arch == tt::ARCH::WORMHOLE_B0 && num_cores > WH_B0_MM_MAX_CORES_NO_STAGGER) {
        mm_kernel_defines["MM_STAGGER_ODD_ROWS"] = "1";
        log_warning(tt::LogOp, "Stagger enabled for matmul op using {} cores.", num_cores);
    }
}

}  // namespace bmm_op_utils

namespace ttnn {

namespace operations {

namespace matmul {

Matmul create_matmul_struct(
    const Tensor& input_tensor_a, const Tensor& input_tensor_b, const struct Matmul& parameters) {
    auto arch = input_tensor_a.device()->arch();
    const bool has_user_grid = parameters.user_core_coord.has_value();
    const bool has_program_config = parameters.program_config.has_value();
    const auto increase_fidelity = !has_program_config && !has_user_grid;
    auto math_fidelity = increase_fidelity ? MathFidelity::HiFi2 : MathFidelity::LoFi;
    auto kernel_config_val = init_device_compute_kernel_config(arch, parameters.compute_kernel_config, math_fidelity);
    bool broadcast_batch =
        parameters.bcast_batch.value_or(get_broadcast_batch(input_tensor_a, input_tensor_b, parameters.program_config));
    TT_FATAL(!(has_user_grid && has_program_config), "Cannot use both user core grid/coordinates and a program config");

    return Matmul{
        parameters.program_config,
        broadcast_batch,
        parameters.output_mem_config,
        parameters.output_dtype.value_or(input_tensor_a.get_dtype()),
        kernel_config_val,
        parameters.untilize_out,
        parameters.user_core_coord,
        parameters.user_fused_activation,
        parameters.user_run_batched,
        parameters.transpose_a,
        parameters.transpose_b,
        parameters.output_tile};
}

Tensor matmul(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const std::optional<const Tensor> bias,
    const struct Matmul& parameters,
    const uint8_t queue_id) {
    std::vector<std::optional<const Tensor>> optional_input_tensors = {};
    std::vector<Tensor> output_tensors;
    if (bias.has_value()) {
        optional_input_tensors.push_back(bias.value());
        output_tensors = {
            Tensor(operation::get_workers_for_op_output({input_tensor_a, input_tensor_b}, {bias.value()}))};
    } else {
        optional_input_tensors.push_back(std::nullopt);
        output_tensors = {Tensor(operation::get_workers_for_op_output({input_tensor_a, input_tensor_b}))};
    }

    operation::launch_op(
        [parameters, queue_id](
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
            const auto& input_tensor_a = input_tensors.at(0);
            const auto& input_tensor_b = input_tensors.at(1);

            return operation::run(
                create_matmul_struct(input_tensor_a, input_tensor_b, parameters),
                {input_tensor_a, input_tensor_b},
                optional_input_tensors,
                {},
                queue_id);
        },
        {input_tensor_a, input_tensor_b},
        output_tensors,
        optional_input_tensors);
    return output_tensors.at(0);
}

void Matmul::validate(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors) const {
    TT_FATAL(input_tensors.size() == 2, "Error");
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);
    const auto& a_shape = input_tensor_a.get_shape();
    const auto& b_shape = input_tensor_b.get_shape();
    auto in0_tile_shape = input_tensor_a.get_tile().get_tile_shape();
    auto in1_tile_shape = input_tensor_b.get_tile().get_tile_shape();

    if (input_tensor_a.device()->arch() == tt::ARCH::GRAYSKULL) {
        TT_FATAL(
            (input_tensor_a.get_tile().get_tile_shape()[1] == TILE_WIDTH && input_tensor_a.get_tile().get_tile_shape()[0] == TILE_HEIGHT),
            "Grayskull does not support tiny tile");
        TT_FATAL(
            (input_tensor_b.get_tile().get_tile_shape()[1] == TILE_WIDTH && input_tensor_b.get_tile().get_tile_shape()[0] == TILE_HEIGHT),
            "Grayskull does not support tiny tile");
    }

    TT_FATAL(
        (input_tensor_a.get_tile().get_tile_shape()[1] == TILE_WIDTH && input_tensor_b.get_tile().get_tile_shape()[0] == TILE_WIDTH),
        "Input tile dims must have inner dim equal to 32 due to llk constraints");

    TT_FATAL(
        (input_tensor_a.get_layout() == Layout::TILE && input_tensor_b.get_layout() == Layout::TILE),
        "Inputs to matmul must be tilized");
    TT_FATAL(
        a_shape[-1] == b_shape[-2],
        "The width of the first tensor must be equal to the height of the second tensor. Mismatch: width={} height={}",
        a_shape[-1],
        b_shape[-2]);

    TT_FATAL(this->bcast_batch.has_value(), "Error");
    if (this->bcast_batch.value()) {
        TT_FATAL(
            get_batch_size(b_shape) == 1,
            "matmul (batch bcast variant) expects input tensors of shapes BCMK*11KN=BCMN or equivalent");
    } else {
        // same condition as above, different message
        TT_FATAL(a_shape.rank() == b_shape.rank() && "bmm (non-bcast matmul) expects input tensors of the same rank", "Error");
        for (auto i = 0; i < a_shape.rank() - 2; i++) {
            TT_FATAL(
                a_shape[i] == b_shape[i],
                "bmm (non-bcast matmul) expects input tensors of shapes BCMK*BCKN=BCMN or equivalent");
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

    MatmulProgramConfig chosen_program_config = get_program_config(input_tensor_a, input_tensor_b, this);

    TT_FATAL(optional_input_tensors.size() == 1, "Error");
    const auto& optional_bias = optional_input_tensors.at(0);
    if (optional_bias.has_value()) {
        TT_FATAL(
            (optional_bias->get_tile().get_tile_shape()[0] == input_tensor_a.get_tile().get_tile_shape()[0] &&
            optional_bias->get_tile().get_tile_shape()[1] == input_tensor_b.get_tile().get_tile_shape()[1]),
            "Input tile dims must have inner dim equal to 32 due to llk constraints");
        const auto& bias = optional_bias.value();
        TT_FATAL(bias.get_layout() == Layout::TILE, "Unsupported input layout");
        const auto& bias_shape = bias.get_shape();
        uint32_t bias_batch_size = get_batch_size(bias_shape);
        TT_FATAL(bias_batch_size == 1, "Unsupported bias shape: batch size not equal to 1.");
        TT_FATAL(
            bias_shape.with_tile_padding()[-2] == in0_tile_shape[0], "Unsupported bias shape: padded second last dimension of bias, {}, not equal to tile height, {}", bias_shape.with_tile_padding()[-2], in0_tile_shape[0]);
        TT_FATAL(
            bias_shape.with_tile_padding()[-1] == b_shape.with_tile_padding()[-1],
            "Unsupported bias shape: padded last dimension of bias, {}, not equal to second input's padded last dimension, {}.", bias_shape.with_tile_padding()[-1], b_shape.with_tile_padding()[-1]);
        TT_FATAL(
            bias_shape[-1] >= b_shape[-1],
            "Unsupported bias shape: last dimension of bias, {}, not equal to or greater than second input's last dimension, {}.", bias_shape[-1], b_shape[-1]);
    }

    if (this->untilize_out) {
        TT_FATAL(this->output_dtype.has_value(), "Error");
        TT_FATAL(
            (this->output_dtype.value() == DataType::BFLOAT16) || (this->output_dtype.value() == DataType::FLOAT32),
            "Unsupported data type: {}", this->output_dtype.value());
    }

    std::visit(
        [input_tensor_a, input_tensor_b, in0_tile_shape, in1_tile_shape, this](const auto& program_config) {
            using ProgramConfigType = std::decay_t<decltype(program_config)>;
            // TODO: For 1D and 2D mcasts, we don't check if tensor is single core or single row/col
            // We can uplift these variants to skip mcasting to support single core (1D) or single row/col (2D)
            if constexpr (std::is_same_v<ProgramConfigType, MatmulMultiCoreReuseMultiCast1DProgramConfig>) {
                if (program_config.mcast_in0) {
                    if (input_tensor_a.is_sharded()) {
                        TT_FATAL(program_config.fuse_batch, "Error");
                        TT_FATAL(input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::WIDTH_SHARDED, "Error");
                        if (this->output_mem_config.is_sharded()) {
                            TT_FATAL(input_tensor_a.memory_config().buffer_type == this->output_mem_config.buffer_type, "Error");
                            TT_FATAL(
                                input_tensor_a.memory_config().memory_layout == this->output_mem_config.memory_layout, "Error");
                        }
                        TT_FATAL(input_tensor_a.shard_spec().value().orientation == ShardOrientation::ROW_MAJOR, "Error");
                        uint32_t M =
                            (program_config.fuse_batch ? input_tensor_a.volume() / input_tensor_a.get_legacy_shape()[-1]
                                                       : input_tensor_a.get_legacy_shape()[-2]) / in0_tile_shape[0];
                        uint32_t N = input_tensor_b.get_legacy_shape()[-1] / in1_tile_shape[1];
                        uint32_t K = input_tensor_a.get_legacy_shape()[-1] / in0_tile_shape[1];
                        uint32_t per_core_M = program_config.per_core_M;
                        uint32_t per_core_N = program_config.per_core_N;
                        auto shard_shape = input_tensor_a.shard_spec().value().shape;

                        // No padding
                        TT_FATAL(M == per_core_M, "Error");
                        TT_FATAL(per_core_M == (shard_shape[0] / in0_tile_shape[0]), "Error");
                        TT_FATAL(K % program_config.in0_block_w == 0, "Error");
                        TT_FATAL((shard_shape[1] / in0_tile_shape[1]) % program_config.in0_block_w == 0, "Error");
                    }
                    if (this->output_mem_config.is_sharded()) {
                        TT_FATAL(this->output_mem_config.memory_layout == TensorMemoryLayout::WIDTH_SHARDED, "Error");
                        uint32_t M =
                            (program_config.fuse_batch ? input_tensor_a.volume() / input_tensor_a.get_legacy_shape()[-1]
                                                       : input_tensor_a.get_legacy_shape()[-2]) /
                            in0_tile_shape[0];
                        uint32_t N = input_tensor_b.get_legacy_shape()[-1] / in1_tile_shape[1];
                        uint32_t per_core_M = program_config.per_core_M;
                        uint32_t per_core_N = program_config.per_core_N;

                        // No padding
                        TT_FATAL(M == per_core_M, "Error");

                        TT_FATAL(program_config.out_subblock_w == per_core_N || program_config.out_subblock_h == 1, "Error");
                    }
                } else {
                    if (input_tensor_a.memory_config().is_sharded()) {
                        TT_FATAL(program_config.fuse_batch, "Error");
                        TT_FATAL(input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED, "Error");
                        if (this->output_mem_config.is_sharded()) {
                            TT_FATAL(input_tensor_a.memory_config().buffer_type == this->output_mem_config.buffer_type, "Error");
                            TT_FATAL(
                                input_tensor_a.memory_config().memory_layout == this->output_mem_config.memory_layout, "Error");
                        }
                        TT_FATAL(input_tensor_a.shard_spec().value().orientation == ShardOrientation::ROW_MAJOR, "Error");
                        uint32_t M =
                            (program_config.fuse_batch ? input_tensor_a.volume() / input_tensor_a.get_legacy_shape()[-1]
                                                       : input_tensor_a.get_legacy_shape()[-2]) / in0_tile_shape[0];
                        uint32_t K = input_tensor_a.get_legacy_shape()[-1] / in0_tile_shape[1];
                        uint32_t per_core_M = program_config.per_core_M;
                        auto shard_shape = input_tensor_a.shard_spec().value().shape;

                        TT_FATAL(div_up(M, per_core_M) == input_tensor_a.shard_spec().value().grid.num_cores(), "Error");
                        TT_FATAL(per_core_M == (shard_shape[0] / in0_tile_shape[0]), "Error");
                        TT_FATAL(K % program_config.in0_block_w == 0, "Error");
                        TT_FATAL(K == (shard_shape[1] / in0_tile_shape[1]), "Error");
                    }
                    if (this->output_mem_config.is_sharded()) {
                        TT_FATAL(this->output_mem_config.memory_layout == TensorMemoryLayout::HEIGHT_SHARDED, "Error");
                        uint32_t M =
                            (program_config.fuse_batch ? input_tensor_a.volume() / input_tensor_a.get_legacy_shape()[-1]
                                                       : input_tensor_a.get_legacy_shape()[-2]) / in0_tile_shape[0];
                        uint32_t N = input_tensor_b.get_legacy_shape()[-1] / in1_tile_shape[1];
                        uint32_t per_core_M = program_config.per_core_M;
                        uint32_t per_core_N = program_config.per_core_N;

                        TT_FATAL(N == per_core_N, "Error");
                        TT_FATAL(program_config.out_subblock_w == per_core_N || program_config.out_subblock_h == 1, "Error");
                    }
                }
                TT_FATAL(input_tensor_b.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED, "Error");
            } else if constexpr (std::is_same_v<
                                     ProgramConfigType,
                                     MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig>) {
                TT_FATAL(input_tensor_a.is_sharded(), "Error");
                TT_FATAL(this->output_mem_config.is_sharded(), "Error");
                TT_FATAL(input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::WIDTH_SHARDED, "Error");
                TT_FATAL(input_tensor_a.memory_config().buffer_type == this->output_mem_config.buffer_type, "Error");
                TT_FATAL(input_tensor_a.memory_config().memory_layout == this->output_mem_config.memory_layout, "Error");
                TT_FATAL(input_tensor_a.shard_spec().value().orientation == ShardOrientation::ROW_MAJOR, "Error");
                uint32_t M = input_tensor_a.volume() / input_tensor_a.get_legacy_shape()[-1] / in0_tile_shape[0];
                uint32_t N = input_tensor_b.get_legacy_shape()[-1] / in1_tile_shape[1];
                uint32_t K = input_tensor_a.get_legacy_shape()[-1] / in0_tile_shape[1];
                uint32_t per_core_M = program_config.per_core_M;
                uint32_t per_core_N = program_config.per_core_N;
                auto shard_shape = input_tensor_a.shard_spec().value().shape;

                // No padding
                TT_FATAL(M == per_core_M, "Error");
                TT_FATAL(per_core_M == (shard_shape[0] / in0_tile_shape[0]), "Error");
                TT_FATAL(K % program_config.in0_block_w == 0, "Error");
                TT_FATAL((shard_shape[1] / in0_tile_shape[1]) % program_config.in0_block_w == 0, "Error");

                // tensor in1
                TT_FATAL(input_tensor_b.memory_config().memory_layout == TensorMemoryLayout::WIDTH_SHARDED, "Error");
            } else if constexpr (std::is_same_v<ProgramConfigType, MatmulMultiCoreReuseMultiCastProgramConfig>) {
                if (input_tensor_a.memory_config().is_sharded()) {
                    auto tensor_a_memory_layout = input_tensor_a.memory_config().memory_layout;
                    uint32_t M = input_tensor_a.volume() / input_tensor_a.get_legacy_shape()[-1] / in0_tile_shape[0];
                    uint32_t K = input_tensor_a.get_legacy_shape()[-1] / in0_tile_shape[1];
                    uint32_t N = input_tensor_b.get_legacy_shape()[-1] / in1_tile_shape[1];
                    uint32_t per_core_M = program_config.per_core_M;
                    auto shard_shape = input_tensor_a.shard_spec().value().shape;

                    TT_FATAL(
                        tensor_a_memory_layout == TensorMemoryLayout::BLOCK_SHARDED ||
                        tensor_a_memory_layout == TensorMemoryLayout::HEIGHT_SHARDED,
                        "Unsupported memory layout {}.",
                        tensor_a_memory_layout);

                    if (tensor_a_memory_layout == TensorMemoryLayout::BLOCK_SHARDED) {
                        if (program_config.transpose_mcast) {
                            TT_FATAL(input_tensor_a.shard_spec().value().orientation == ShardOrientation::COL_MAJOR, "Error");
                        } else {
                            TT_FATAL(input_tensor_a.shard_spec().value().orientation == ShardOrientation::ROW_MAJOR, "Error");
                        }
                        if (this->output_mem_config.is_sharded()) {
                            TT_FATAL(input_tensor_a.memory_config().buffer_type == this->output_mem_config.buffer_type, "Error");
                            TT_FATAL(
                                input_tensor_a.memory_config().memory_layout == this->output_mem_config.memory_layout, "Error");
                        }

                    } else if (tensor_a_memory_layout == TensorMemoryLayout::HEIGHT_SHARDED) {
                        TT_FATAL(!program_config.transpose_mcast, "Error");
                        TT_FATAL(K == program_config.in0_block_w, "Error");
                        TT_FATAL(program_config.in0_block_w == (shard_shape[1] / in0_tile_shape[1]), "Error");
                        TT_FATAL(
                            input_tensor_a.shard_spec()->grid.bounding_box().start_coord.x ==
                            input_tensor_a.shard_spec()->grid.bounding_box().end_coord.x, "Error");
                    }

                    TT_FATAL(per_core_M == (shard_shape[0] / in0_tile_shape[0]), "Error");
                    TT_FATAL((shard_shape[1] / in0_tile_shape[1]) % program_config.in0_block_w == 0, "Error");
                }

                if (input_tensor_b.memory_config().is_sharded()) {
                    TT_FATAL(!program_config.transpose_mcast, "Error");
                    auto tensor_b_memory_layout = input_tensor_b.memory_config().memory_layout;
                    TT_FATAL(tensor_b_memory_layout == TensorMemoryLayout::WIDTH_SHARDED, "Error");
                    if (input_tensor_b.buffer()->buffer_type() != tt_metal::BufferType::DRAM) {
                        TT_FATAL(
                            program_config.per_core_N == (input_tensor_b.shard_spec().value().shape[1] / in1_tile_shape[1]), "Error");
                    }
                    TT_FATAL(
                        input_tensor_b.shard_spec()->grid.bounding_box().start_coord.y ==
                        input_tensor_b.shard_spec()->grid.bounding_box().end_coord.y, "Error");
                }

                if (this->output_mem_config.is_sharded()) {
                    TT_FATAL(this->output_mem_config.memory_layout == TensorMemoryLayout::BLOCK_SHARDED, "Error");
                    uint32_t M = input_tensor_a.volume() / input_tensor_a.get_legacy_shape()[-1] / in0_tile_shape[0];
                    uint32_t N = input_tensor_b.get_legacy_shape()[-1] / in1_tile_shape[1];
                    uint32_t per_core_M = program_config.per_core_M;
                    uint32_t per_core_N = program_config.per_core_N;

                    TT_FATAL(program_config.out_subblock_w == per_core_N || program_config.out_subblock_h == 1, "Error");
                }
            } else if constexpr (std::is_same_v<ProgramConfigType, MatmulMultiCoreReuseProgramConfig>) {
                uint32_t M = input_tensor_a.get_legacy_shape()[-2] / in0_tile_shape[0];
                uint32_t total_M = input_tensor_a.volume() / input_tensor_a.get_legacy_shape()[-1] / in0_tile_shape[0];
                uint32_t N = input_tensor_b.get_legacy_shape()[-1] / in1_tile_shape[1];
                uint32_t K = input_tensor_a.get_legacy_shape()[-1];
                uint32_t per_core_M = program_config.per_core_M;
                uint32_t per_core_N = program_config.per_core_N;
                if (per_core_M > M) {
                    TT_FATAL(per_core_M % M == 0, "per_core_M must be a multiple of M if per_core_M > M!");
                    TT_FATAL(total_M % per_core_M == 0, "input a total height must be divisible by per_core_M!");
                } else {
                    TT_FATAL(M % per_core_M == 0, "per_core_M must divide M if per_core_M < M!");
                }
                TT_FATAL(N == per_core_N, "Error");
                if (input_tensor_a.is_sharded()) {
                    TT_FATAL(input_tensor_a.memory_config().memory_layout != TensorMemoryLayout::WIDTH_SHARDED, "Error");
                    auto in0_shard_shape = input_tensor_a.shard_spec().value().shape;

                    TT_FATAL(K == in0_shard_shape[1], "Error");
                    TT_FATAL(in0_shard_shape[1] == program_config.in0_block_w * in0_tile_shape[1], "Error");
                    TT_FATAL(per_core_M * in0_tile_shape[0] == in0_shard_shape[0], "Error");

                    if (input_tensor_b.is_sharded()) {
                        TT_FATAL(
                            input_tensor_a.memory_config().buffer_type == input_tensor_b.memory_config().buffer_type, "Error");
                        TT_FATAL(
                            input_tensor_a.memory_config().memory_layout ==
                            input_tensor_b.memory_config().memory_layout, "Error");
                        TT_FATAL(input_tensor_a.shard_spec().value().grid == input_tensor_b.shard_spec().value().grid, "Error");
                        TT_FATAL(
                            input_tensor_a.shard_spec().value().orientation ==
                            input_tensor_b.shard_spec().value().orientation, "Error");
                    }
                    if (this->output_mem_config.is_sharded()) {
                        TT_FATAL(input_tensor_a.memory_config().buffer_type == this->output_mem_config.buffer_type, "Error");
                        TT_FATAL(input_tensor_a.memory_config().memory_layout == this->output_mem_config.memory_layout, "Error");
                    }
                }

                uint32_t batch_size_a = get_batch_size(input_tensor_a.get_legacy_shape());
                uint32_t batch_size_b = get_batch_size(input_tensor_b.get_legacy_shape());
                bool broadcast_batch = batch_size_a > 1 and batch_size_b == 1;
                TT_FATAL(!broadcast_batch, "Error");

                if (input_tensor_b.is_sharded()) {
                    TT_FATAL(per_core_M % M == 0, "per_core_M must be a multiple of M if input b is sharded!");
                    TT_FATAL(input_tensor_b.memory_config().memory_layout != TensorMemoryLayout::WIDTH_SHARDED, "Error");
                    auto in1_shard_shape = input_tensor_b.shard_spec().value().shape;
                    TT_FATAL(in1_shard_shape[1] == input_tensor_b.get_legacy_shape()[-1], "Error");
                    TT_FATAL(per_core_N * in1_tile_shape[1] == in1_shard_shape[1], "Error");
                    TT_FATAL(in1_shard_shape[0] % K == 0, "Error");
                }
                if (this->output_mem_config.is_sharded()) {
                    TT_FATAL(this->output_mem_config.memory_layout != TensorMemoryLayout::WIDTH_SHARDED, "Error");
                    TT_FATAL(program_config.out_subblock_w == per_core_N || program_config.out_subblock_h == 1, "Error");
                }
            } else {
                TT_FATAL(input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED, "Error");
                TT_FATAL(input_tensor_b.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED, "Error");
                TT_FATAL(this->output_mem_config.memory_layout == TensorMemoryLayout::INTERLEAVED, "Error");
            }
            if constexpr (
                std::is_same_v<ProgramConfigType, MatmulMultiCoreReuseProgramConfig> ||
                std::is_same_v<ProgramConfigType, MatmulMultiCoreReuseMultiCastProgramConfig> ||
                std::is_same_v<ProgramConfigType, MatmulMultiCoreReuseMultiCast1DProgramConfig>) {
                TT_FATAL(
                    (input_tensor_a.get_legacy_shape()[-1] / in0_tile_shape[1]) % program_config.in0_block_w == 0,
                    "Kt must be divisible by in0_block_w");
                TT_FATAL(
                    program_config.per_core_M % program_config.out_subblock_h == 0,
                    "per_core_M must be divisible by out_subblock_h");
                TT_FATAL(
                    program_config.per_core_N % program_config.out_subblock_w == 0,
                    "per_core_N must be divisible by out_subblock_w");
            }
        },
        chosen_program_config);
}

std::vector<tt::tt_metal::LegacyShape> Matmul::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    const tt::tt_metal::LegacyShape& input_shape_a = input_tensors.at(0).get_legacy_shape();
    const tt::tt_metal::LegacyShape& input_shape_b = input_tensors.at(1).get_legacy_shape();
    const uint32_t a_rank = input_shape_a.rank();
    const uint32_t b_rank = input_shape_b.rank();
    const uint32_t out_rank = std::max(a_rank, b_rank);
    const uint32_t rank_difference = out_rank - a_rank;
    tt::tt_metal::LegacyShape output_shape = (b_rank > a_rank) ? input_shape_b : input_shape_a;
    auto dimensions_pads = std::vector<Padding::PadDimension>();

    for (auto index = 0; index < rank_difference; index++) {
        TT_FATAL(input_shape_b[index] == 1, "When in1 rank greater than in0 rank front dimensions need to be 1");
        output_shape[index] = input_shape_b[index];
        dimensions_pads.push_back(input_shape_b.padding()[index]);
    }
    for (auto index = 0; index < a_rank - 1; index++) {
        output_shape[rank_difference + index] = input_shape_a[index];
        dimensions_pads.push_back(input_shape_a.padding()[index]);
    }
    output_shape[-1] = input_shape_b[-1];
    dimensions_pads.push_back(input_shape_b.padding()[b_rank - 1]);
    const auto padding = Padding(dimensions_pads, Padding::PadValue::Any);
    return {tt::tt_metal::LegacyShape(output_shape, padding)};
}

std::vector<Tensor> Matmul::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);
    auto in0_tile_shape = input_tensor_a.get_tile().get_tile_shape();
    auto in1_tile_shape = input_tensor_b.get_tile().get_tile_shape();
    if (this->output_tile.has_value()) {
        TT_FATAL(this->output_tile->get_tile_shape()[1] % in1_tile_shape[1] == 0, "the override output tile width be multiple of in1 tile width");
        TT_FATAL(this->output_tile->get_tile_shape()[0] == in0_tile_shape[0], "the override output tile height must equal to the in0 tile height");
        if (this->output_tile->get_tile_shape()[1] != in1_tile_shape[1]) {
            TT_FATAL(this->output_tile->get_tile_shape()[0] <= constants::FACE_HEIGHT, "the override output tile height must equal or less to face height");
        }
        if (!this->output_mem_config.is_sharded()) {
            TT_FATAL(this->output_tile->get_tile_shape()[1] == in1_tile_shape[1], "the override output tile width must equal to the in0 tile width");
        }
    }
    auto output_tile = this->output_tile.value_or(tt::tt_metal::Tile({in0_tile_shape[0], in1_tile_shape[1]}));
    auto tile_width_ratio = output_tile.get_tile_shape()[1] / in1_tile_shape[1];
    auto output_layout = this->untilize_out ? Layout::ROW_MAJOR : Layout::TILE;
    TT_FATAL(this->output_dtype.has_value(), "Error");
    if (this->output_mem_config.is_sharded()) {
        MatmulProgramConfig chosen_program_config = get_program_config(input_tensor_a, input_tensor_b, this);
        return std::visit(
            [&](const auto& program_config) -> std::vector<Tensor> {
                using ProgramConfigType = std::decay_t<decltype(program_config)>;
                if constexpr (std::is_same_v<ProgramConfigType, MatmulMultiCoreReuseMultiCast1DProgramConfig>) {
                    uint32_t M =
                        (program_config.fuse_batch ? input_tensor_a.volume() / input_tensor_a.get_legacy_shape()[-1]
                                                   : input_tensor_a.get_legacy_shape()[-2]) / in0_tile_shape[0];
                    uint32_t N = input_tensor_b.get_legacy_shape()[-1] / in1_tile_shape[1];
                    uint32_t per_core_M = program_config.per_core_M;
                    uint32_t per_core_N = program_config.per_core_N;

                    TT_FATAL(per_core_N % tile_width_ratio == 0, "per_core_N must be divisible by override output tile width");

                    uint32_t num_blocks_y = (M - 1) / per_core_M + 1;
                    uint32_t num_blocks_x = (N - 1) / per_core_N + 1;
                    uint32_t num_blocks_total = num_blocks_y * num_blocks_x;
                    uint32_t num_cores = num_blocks_x * num_blocks_y;
                    CoreRangeSet all_cores =
                        num_cores_to_corerange_set(num_cores, program_config.compute_with_storage_grid_size, true);
                    ShardSpec shard_spec = ShardSpec{
                        all_cores, {per_core_M * in0_tile_shape[0], per_core_N * in1_tile_shape[1]}, ShardOrientation::ROW_MAJOR};
                    auto mem_config = this->output_mem_config;
                    mem_config.shard_spec = shard_spec;
                    return {create_device_tensor(
                        this->compute_output_shapes(input_tensors).at(0),
                        this->output_dtype.value(),
                        output_layout,
                        input_tensor_a.device(),
                        mem_config,
                        output_tile)};
                } else if constexpr (std::is_same_v<
                                         ProgramConfigType,
                                         MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig>) {
                    uint32_t M = input_tensor_a.volume() / input_tensor_a.get_legacy_shape()[-1];
                    uint32_t N = input_tensor_b.get_legacy_shape()[-1] / in1_tile_shape[1];
                    auto input_tensor_b_shape = input_tensor_b.get_legacy_shape();

                    uint32_t per_core_M = program_config.per_core_M;
                    uint32_t per_core_N = program_config.per_core_N;

                    TT_FATAL(per_core_N % tile_width_ratio == 0, "per_core_N must be divisible by override output tile width");

                    CoreRangeSet all_cores = input_tensor_a.shard_spec().value().grid;
                    ShardSpec shard_spec = ShardSpec{
                        all_cores, {per_core_M * in0_tile_shape[0], per_core_N * in1_tile_shape[1]}, ShardOrientation::ROW_MAJOR};
                    auto mem_config = this->output_mem_config;
                    mem_config.shard_spec = shard_spec;
                    return {create_device_tensor(
                        this->compute_output_shapes(input_tensors).at(0),
                        this->output_dtype.value(),
                        output_layout,
                        input_tensor_a.device(),
                        mem_config,
                        output_tile)};
                } else if constexpr (std::is_same_v<ProgramConfigType, MatmulMultiCoreReuseMultiCastProgramConfig>) {
                    uint32_t M = input_tensor_a.volume() / input_tensor_a.get_legacy_shape()[-1] / in0_tile_shape[0];
                    uint32_t N = input_tensor_b.get_legacy_shape()[-1] / in1_tile_shape[1];
                    uint32_t per_core_M = program_config.per_core_M;
                    uint32_t per_core_N = program_config.per_core_N;

                    TT_FATAL(per_core_N % tile_width_ratio == 0, "per_core_N must be divisible by override output tile width");

                    uint32_t num_blocks_y = (M - 1) / per_core_M + 1;
                    uint32_t num_blocks_x = (N - 1) / per_core_N + 1;
                    uint32_t num_blocks_total = num_blocks_y * num_blocks_x;
                    uint32_t num_cores = num_blocks_x * num_blocks_y;
                    CoreRangeSet all_cores({});
                    ShardOrientation shard_orientation;
                    if (program_config.transpose_mcast) {
                        all_cores = CoreRangeSet({CoreRange({0, 0}, {num_blocks_y - 1, num_blocks_x - 1})});
                        shard_orientation = ShardOrientation::COL_MAJOR;
                    } else {
                        all_cores = CoreRangeSet({CoreRange({0, 0}, {num_blocks_x - 1, num_blocks_y - 1})});
                        shard_orientation = ShardOrientation::ROW_MAJOR;
                    }
                    ShardSpec shard_spec =
                        ShardSpec{all_cores, {per_core_M * in0_tile_shape[0], per_core_N * in1_tile_shape[1]}, shard_orientation};
                    auto mem_config = this->output_mem_config;
                    mem_config.shard_spec = shard_spec;
                    return {create_device_tensor(
                        this->compute_output_shapes(input_tensors).at(0),
                        this->output_dtype.value(),
                        output_layout,
                        input_tensor_a.device(),
                        mem_config,
                        output_tile)};
                } else if constexpr (std::is_same_v<ProgramConfigType, MatmulMultiCoreReuseProgramConfig>) {
                    uint32_t M = input_tensor_a.volume() / input_tensor_a.get_legacy_shape()[-1] / in0_tile_shape[0];
                    uint32_t N = input_tensor_b.get_legacy_shape()[-1] / in1_tile_shape[1];
                    uint32_t per_core_M = program_config.per_core_M;
                    uint32_t per_core_N = program_config.per_core_N;

                    TT_FATAL(per_core_N % tile_width_ratio == 0, "per_core_N must be divisible by override output tile width");

                    uint32_t num_blocks_y = (M - 1) / per_core_M + 1;
                    uint32_t num_blocks_x = (N - 1) / per_core_N + 1;
                    uint32_t num_blocks_total = num_blocks_y * num_blocks_x;
                    uint32_t num_cores = num_blocks_x * num_blocks_y;
                    ShardOrientation shard_orientation = ShardOrientation::COL_MAJOR;
                    if (input_tensor_a.is_sharded()) {
                        shard_orientation = input_tensor_a.shard_spec().value().orientation;
                    } else if (input_tensor_b.is_sharded()) {
                        shard_orientation = input_tensor_b.shard_spec().value().orientation;
                    }

                    CoreRangeSet all_cores = num_cores_to_corerange_set(
                        num_cores,
                        program_config.compute_with_storage_grid_size,
                        shard_orientation == ShardOrientation::ROW_MAJOR);
                    ShardSpec shard_spec =
                        ShardSpec{all_cores, {per_core_M * in0_tile_shape[0], per_core_N * in1_tile_shape[1]}, shard_orientation};
                    auto mem_config = this->output_mem_config;
                    mem_config.shard_spec = shard_spec;
                    return {create_device_tensor(
                        this->compute_output_shapes(input_tensors).at(0),
                        this->output_dtype.value(),
                        output_layout,
                        input_tensor_a.device(),
                        mem_config,
                        output_tile)};
                } else {
                    TT_FATAL(in0_tile_shape[0] == TILE_HEIGHT and in0_tile_shape[1] == TILE_WIDTH,
                                "matmul with non-optimized program config does not support tiny tile");
                    TT_FATAL(in1_tile_shape[0] == TILE_HEIGHT and in1_tile_shape[1] == TILE_WIDTH,
                                "matmul with non-optimized program config does not support tiny tile");
                    if (this->output_tile.has_value()) {
                        TT_FATAL(this->output_tile->get_tile_shape()[0] == TILE_HEIGHT and this->output_tile->get_tile_shape()[1] == TILE_WIDTH,
                                    "matmul with non-optimized program config does not support tiny tile");
                    }
                    TT_THROW("Unsupported op for output sharding");
                    return {};
                }
            },
            chosen_program_config);
    }

    return operation::generic_create_output_tensors(
        *this, input_tensors, this->output_dtype.value(), Layout::TILE, this->output_mem_config, output_tile);
}

operation::ProgramWithCallbacks Matmul::create_program(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);
    const auto& bias = optional_input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);

    TT_FATAL(this->output_dtype.has_value(), "Error");
    tt::tt_metal::DataType output_dtype = this->output_dtype.value();

    bool fuse_batch = true;
    // TODO: If input_tensor_a.get_legacy_shape()[0] * input_tensor_a.get_legacy_shape()[1] * ... except last two
    // dimensions == 1, does matmuls work if we treat it as bmm
    // TODO: Only for MatmulMultiCoreReuseProgramConfig we allow this as single core matmul/bmm
    TT_FATAL(this->compute_kernel_config.has_value(), "Error");
    TT_FATAL(this->bcast_batch.has_value(), "Error");
    bool broadcast_batch = this->bcast_batch.value();

    MatmulProgramConfig chosen_program_config = get_program_config(input_tensor_a, input_tensor_b, this);

    return std::visit(
        [&](const auto& program_config) -> operation::ProgramWithCallbacks {
            using ProgramConfigType = std::decay_t<decltype(program_config)>;
            if constexpr (std::is_same_v<ProgramConfigType, MatmulMultiCoreReuseProgramConfig>) {
                TT_FATAL(!bias.has_value(), "Bias is not supported for MatmulMultiCoreReuseProgramConfig!");
                // TODO: fuse_batch doesn't do anything for this variant! Code is doing fuse_batch=false
                return bmm_multi_core_reuse_optimized(
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
            } else if constexpr (std::is_same_v<ProgramConfigType, MatmulMultiCoreReuseMultiCastProgramConfig>) {
                return matmul_multi_core_reuse_mcast_2d_optimized(
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
                    program_config.per_core_M,
                    program_config.per_core_N,
                    program_config.fuse_batch,
                    program_config.transpose_mcast,
                    program_config.fused_activation,
                    this->untilize_out);
            } else if constexpr (std::is_same_v<ProgramConfigType, MatmulMultiCoreReuseMultiCast1DProgramConfig>) {
                return matmul_multi_core_reuse_mcast_1d_optimized(
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
                    program_config.per_core_M,
                    program_config.per_core_N,
                    program_config.fuse_batch,
                    program_config.fused_activation,
                    program_config.mcast_in0,
                    this->untilize_out);
            } else if constexpr (std::is_same_v<
                                     ProgramConfigType,
                                     MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig>) {
                return matmul_multi_core_reuse_dram_sharded_optimized(
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
            } else if constexpr (std::is_same_v<ProgramConfigType, MatmulMultiCoreNonOptimizedReuseProgramConfig>) {
                TT_FATAL(!bias.has_value(), "Bias is not supported for matmul multi core non-optimized reuse");
                return matmul_multi_core_reuse(input_tensor_a, input_tensor_b, output_tensor, broadcast_batch);
            } else if constexpr (std::is_same_v<ProgramConfigType, MatmulMultiCoreProgramConfig>) {
                TT_FATAL(!bias.has_value(), "Bias is not supported for matmul multi core");
                return matmul_multi_core(input_tensor_a, input_tensor_b, output_tensor, broadcast_batch);
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

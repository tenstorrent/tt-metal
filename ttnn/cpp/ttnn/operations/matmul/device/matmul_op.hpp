// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <optional>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/types.hpp"

#include "ttnn/operations/ccl/ccl_op_fusion.hpp"

namespace ttnn {

namespace operations {

namespace matmul {

using ttnn::operations::unary::UnaryWithParam;
using ttnn::Shape;

/*
 * GENERAL MATMUL AND BMM
 */
operation::ProgramWithCallbacks matmul_multi_core(
    const Tensor &input_tensor_a, const Tensor &input_tensor_b, Tensor &output_tensor, bool bcast_batch);
operation::ProgramWithCallbacks matmul_multi_core_reuse(
    const Tensor &input_tensor_a, const Tensor &input_tensor_b, Tensor &output_tensor, bool bcast_batch);
operation::ProgramWithCallbacks matmul_multi_core_reuse_mcast(
    const Tensor &input_tensor_a, const Tensor &input_tensor_b, Tensor &output_tensor, bool bcast_batch);

operation::ProgramWithCallbacks matmul_multi_core_reuse_mcast_1d_optimized(
    const Tensor &input_tensor_a,
    const Tensor &input_tensor_b,
    const std::optional<const Tensor> bias,
    Tensor &output_tensor,
    bool bcast_batch,
    CoreCoord compute_with_storage_grid_size,
    DeviceComputeKernelConfig compute_kernel_config,
    uint32_t in0_block_w,
    uint32_t out_subblock_h,
    uint32_t out_subblock_w,
    uint32_t per_core_M,
    uint32_t per_core_N,
    bool fuse_batch,
    std::optional<UnaryWithParam> fused_activation,
    bool mcast_in0,
    bool untilize_out);
operation::ProgramWithCallbacks matmul_multi_core_reuse_dram_sharded_optimized(
    const Tensor &input_tensor_a,
    const Tensor &input_tensor_b,
    const std::optional<const Tensor> bias,
    Tensor &output_tensor,
    DeviceComputeKernelConfig compute_kernel_config,
    uint32_t in0_block_w,
    uint32_t per_core_M,
    uint32_t per_core_N,
    std::optional<UnaryWithParam> fused_activation,
    bool untilize_out,
    bool skip_compute,
    bool skip_in0_mcast,
    bool skip_write_back);
operation::ProgramWithCallbacks matmul_multi_core_reuse_mcast_2d_optimized(
    const Tensor &input_tensor_a,
    const Tensor &input_tensor_b,
    const std::optional<const Tensor> bias,
    Tensor &output_tensor,
    bool bcast_batch,
    CoreCoord compute_with_storage_grid_size,
    DeviceComputeKernelConfig compute_kernel_config,
    uint32_t in0_block_w,
    uint32_t out_subblock_h,
    uint32_t out_subblock_w,
    uint32_t per_core_M,
    uint32_t per_core_N,
    bool fuse_batch,
    bool transpose_mcast,
    std::optional<UnaryWithParam> fused_activation,
    bool untilize_out);
operation::ProgramWithCallbacks bmm_multi_core_reuse_optimized(
    const Tensor &input_tensor_a,
    const Tensor &input_tensor_b,
    Tensor &output_tensor,
    bool bcast_batch,
    CoreCoord compute_with_storage_grid_size,
    tt::tt_metal::DataType output_dtype,
    DeviceComputeKernelConfig compute_kernel_config,
    uint32_t in0_block_w,
    uint32_t out_subblock_h,
    uint32_t out_subblock_w,
    uint32_t per_core_M,
    uint32_t per_core_N,
    bool fuse_batch,
    bool untilize_out);

// TODO: Uplift this to support fused activation and bias
// TODO: Uplift this to support bcast batch for in1; currently, only allows B=1 for in1 iff B=1 for in0 (ie. single
// core)
struct MatmulMultiCoreReuseProgramConfig {
    CoreCoord compute_with_storage_grid_size;
    std::size_t in0_block_w;
    std::size_t out_subblock_h;
    std::size_t out_subblock_w;
    std::size_t per_core_M;
    std::size_t per_core_N;
};

struct MatmulMultiCoreReuseMultiCastProgramConfig {
    CoreCoord compute_with_storage_grid_size;
    std::size_t in0_block_w;
    std::size_t out_subblock_h;
    std::size_t out_subblock_w;
    std::size_t per_core_M;
    std::size_t per_core_N;
    bool transpose_mcast;
    std::optional<UnaryWithParam> fused_activation;
    bool fuse_batch = true;
};

struct MatmulMultiCoreReuseMultiCast1DProgramConfig {
    CoreCoord compute_with_storage_grid_size;
    std::size_t in0_block_w;
    std::size_t out_subblock_h;
    std::size_t out_subblock_w;
    std::size_t per_core_M;
    std::size_t per_core_N;
    bool fuse_batch;
    std::optional<UnaryWithParam> fused_activation;
    bool mcast_in0;
};

struct MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig {
    std::size_t in0_block_w;
    std::size_t per_core_M;
    std::size_t per_core_N;
    std::optional<UnaryWithParam> fused_activation;
};

struct MatmulMultiCoreProgramConfig {};

struct MatmulMultiCoreNonOptimizedReuseProgramConfig {};

using MatmulProgramConfig = std::variant<
    MatmulMultiCoreProgramConfig,
    MatmulMultiCoreNonOptimizedReuseProgramConfig,
    MatmulMultiCoreReuseProgramConfig,
    MatmulMultiCoreReuseMultiCastProgramConfig,
    MatmulMultiCoreReuseMultiCast1DProgramConfig,
    MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig>;

struct Matmul {
    const std::optional<const MatmulProgramConfig> program_config = std::nullopt;
    const std::optional<bool> bcast_batch = std::nullopt;
    const MemoryConfig output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG;
    const std::optional<DataType> output_dtype = std::nullopt;
    const std::optional<DeviceComputeKernelConfig> compute_kernel_config = std::nullopt;
    const bool untilize_out = false;
    const std::optional<const CoreCoord> user_core_coord = std::nullopt;
    const std::optional<UnaryWithParam> user_fused_activation = std::nullopt;
    const bool user_run_batched = false;
    const bool transpose_a = false;
    const bool transpose_b = false;
    const std::optional<const Tile> output_tile;

    void validate(
        const std::vector<Tensor> &input_tensors,
        const std::vector<std::optional<const Tensor>> &optional_input_tensors) const;
    std::vector<ttnn::Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<ttnn::Shape> compute_output_shapes_dram_sharded(
        const std::vector<Tensor> &input_tensors, uint32_t N_unpadded) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &input_tensors,
        const std::vector<std::optional<const Tensor>> &optional_input_tensors,
        std::vector<Tensor> &output_tensors) const;
    operation::OpPerformanceModel create_op_performance_model(
        const std::vector<Tensor> &input_tensors,
        const std::vector<std::optional<const Tensor>> &optional_input_tensors,
        std::vector<Tensor> &output_tensors) const;
};

Matmul create_matmul_struct(
    const Tensor &input_tensor_a,
    const Tensor &input_tensor_b,
    const struct Matmul &parameters
);

operation::ProgramWithCallbacks matmul_multi_core_reuse_mcast_1d_optimized_helper(
    tt::tt_metal::Program& program,
    const Tensor &input_tensor_a,
    const Tensor &input_tensor_b,
    const std::optional<const Tensor> bias,
    Tensor &output_tensor,
    bool bcast_batch,
    DeviceComputeKernelConfig compute_kernel_config,
    const MatmulProgramConfig program_config,
    bool untilize_out,
    std::optional<ttnn::experimental::ccl::MatmulFusedOpSignaler> &fused_op_signaler);
operation::ProgramWithCallbacks matmul_multi_core_reuse_mcast_2d_optimized_helper(
    tt::tt_metal::Program& program,
    const Tensor &input_tensor_a,
    const Tensor &input_tensor_b,
    const std::optional<const Tensor> bias,
    Tensor &output_tensor,
    bool bcast_batch,
    DeviceComputeKernelConfig compute_kernel_config,
    const MatmulProgramConfig program_config,
    bool untilize_out,
    std::optional<ttnn::experimental::ccl::MatmulFusedOpSignaler> &matmul_signal_info);

Tensor matmul(
    const Tensor &input_tensor_a,
    const Tensor &input_tensor_b,
    const std::optional<const Tensor> bias = std::nullopt,
    const struct Matmul &parameters = Matmul{},
    const uint8_t queue_id = 0);

}  // namespace matmul

}  // namespace operations

}  // namespace ttnn

namespace bmm_op_utils {

std::tuple<uint32_t, uint32_t> get_matmul_subblock_params(
    const uint32_t per_core_M,
    const uint32_t per_core_N,
    const bool per_core_M_equals_subblock_h_constraint,
    const bool per_core_N_equals_subblock_w_constraint,
    const bool fp32_dest_acc_en);

void add_stagger_defines_if_needed(const tt::ARCH arch, const int num_cores, std::map<string, string>& mm_kernel_defines);

}  // namespace bmm_op_utils

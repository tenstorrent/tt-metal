// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <optional>
#include <tt-metalium/mesh_coord.hpp>
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/types.hpp"

namespace ttnn {

namespace operations {

namespace matmul {

// shared variables between override and program

enum class Matmul1DType { MCAST_IN0, GATHER_IN0, MCAST_IN1 };

struct matmul_mcast_1d_common_override_variables_t {
    std::vector<tt::tt_metal::KernelHandle> kernels;
    std::vector<tt::tt_metal::CBHandle> cbs;
    bool extract_shard_sub_blocks;
    CoreCoord start_core;
    std::vector<CoreCoord> cores;
    uint32_t num_cores_with_work;
    Matmul1DType type;
};

// Define the buffering depth for input CBs (0 and 1) for mcast variants.
// 2 = double buffer, 3 = triple buffer, etc.
// Allows easily changing buffering strategy in one place for relevant factories.
constexpr uint32_t MCAST_INPUT_BUFFERING_DEPTH = 2;

using ttnn::operations::unary::UnaryWithParam;

/**
 * @brief Computes the output shape of a matmul operation given two input tensors
 *
 * Determines the output shape based on the broadcasting rules for matrix multiplication:
 * - For 2D tensors: [m, k] @ [k, n] -> [m, n]
 * - For tensors with batch dimensions, the batch dimensions are broadcast
 * - For vector-matrix multiplication (rank 1 @ rank 2), the result is a vector
 *
 * @param input_tensor_a First input tensor
 * @param input_tensor_b Second input tensor
 * @return Shape of the resulting tensor after matmul
 */
ttnn::Shape compute_matmul_output_shape(const Tensor& input_tensor_a, const Tensor& input_tensor_b);

/*
 * GENERAL MATMUL AND BMM
 */
tt::tt_metal::operation::ProgramWithCallbacks matmul_multi_core(
    const Tensor& input_tensor_a, const Tensor& input_tensor_b, Tensor& output_tensor, bool bcast_batch);
tt::tt_metal::operation::ProgramWithCallbacks matmul_multi_core_reuse(
    const Tensor& input_tensor_a, const Tensor& input_tensor_b, Tensor& output_tensor, bool bcast_batch);
tt::tt_metal::operation::ProgramWithCallbacks matmul_multi_core_reuse_mcast(
    const Tensor& input_tensor_a, const Tensor& input_tensor_b, Tensor& output_tensor, bool bcast_batch);

tt::tt_metal::operation::ProgramWithCallbacks matmul_multi_core_reuse_mcast_1d_optimized(
    const Tensor& input_tensor_a,
    const std::vector<Tensor>& input_tensors_b,
    const std::optional<const Tensor>& bias,
    const std::vector<Tensor>& output_tensors,
    bool bcast_batch,
    CoreCoord compute_with_storage_grid_size,
    DeviceComputeKernelConfig compute_kernel_config,
    uint32_t in0_block_w,
    uint32_t out_subblock_h,
    uint32_t out_subblock_w,
    uint32_t out_block_h,
    uint32_t out_block_w,
    uint32_t per_core_M,
    uint32_t per_core_N,
    bool fuse_batch,
    const std::optional<UnaryWithParam>& fused_activation,
    bool mcast_in0,
    bool gather_in0,
    const CoreRangeSet& hop_cores,
    bool untilize_out,
    const std::optional<const tt::tt_metal::experimental::GlobalCircularBuffer>& global_cb,
    uint32_t num_global_cb_receivers,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id);
tt::tt_metal::operation::ProgramWithCallbacks matmul_multi_core_reuse_dram_sharded_optimized(
    const ttnn::MeshCoordinate& mesh_coord,
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const std::optional<const Tensor>& bias,
    Tensor& output_tensor,
    DeviceComputeKernelConfig compute_kernel_config,
    uint32_t in0_block_w,
    uint32_t per_core_M,
    uint32_t per_core_N,
    std::optional<UnaryWithParam> fused_activation,
    bool untilize_out,
    bool skip_compute,
    bool skip_in0_mcast,
    bool skip_write_back);
tt::tt_metal::operation::ProgramWithCallbacks matmul_multi_core_reuse_mcast_2d_optimized(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const std::optional<const Tensor>& bias,
    Tensor& output_tensor,
    bool bcast_batch,
    CoreCoord compute_with_storage_grid_size,
    DeviceComputeKernelConfig compute_kernel_config,
    uint32_t in0_block_w,
    uint32_t out_subblock_h,
    uint32_t out_subblock_w,
    uint32_t out_block_h,
    uint32_t out_block_w,
    uint32_t per_core_M,
    uint32_t per_core_N,
    bool fuse_batch,
    bool transpose_mcast,
    std::optional<UnaryWithParam> fused_activation,
    bool untilize_out);
tt::tt_metal::operation::ProgramWithCallbacks bmm_multi_core_reuse_optimized(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    Tensor& output_tensor,
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

/*
 * SPARSE MATMUL
 */
tt::tt_metal::operation::ProgramWithCallbacks sparse_bmm_multi_core_reuse(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const Tensor& sparsity,
    uint32_t num_batches,
    Tensor& output_tensor,
    CoreCoord compute_with_storage_grid_size,
    tt::tt_metal::DataType output_dtype,
    DeviceComputeKernelConfig compute_kernel_config,
    uint32_t in0_block_w,
    uint32_t out_subblock_h,
    uint32_t out_subblock_w,
    uint32_t per_core_M,
    uint32_t per_core_N);

// TODO: Uplift this to support fused activation and bias
// TODO: Uplift this to support bcast batch for in1; currently, only allows B=1
// for in1 iff B=1 for in0 (ie. single core)
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
    std::size_t out_block_h;
    std::size_t out_block_w;
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
    std::size_t out_block_h;
    std::size_t out_block_w;
    std::size_t per_core_M;
    std::size_t per_core_N;
    bool fuse_batch;
    std::optional<UnaryWithParam> fused_activation;
    bool mcast_in0;
    bool gather_in0;
    CoreRangeSet hop_cores;
    std::size_t num_global_cb_receivers;
    bool untilize_out;
};

struct MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig {
    std::size_t in0_block_w;
    std::size_t per_core_M;
    std::size_t per_core_N;
    std::optional<UnaryWithParam> fused_activation;
};

struct MatmulMultiCoreProgramConfig {};

using MatmulProgramConfig = std::variant<
    MatmulMultiCoreProgramConfig,
    MatmulMultiCoreReuseProgramConfig,
    MatmulMultiCoreReuseMultiCastProgramConfig,
    MatmulMultiCoreReuseMultiCast1DProgramConfig,
    MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig>;

struct Matmul {
    const std::optional<const MatmulProgramConfig> program_config = std::nullopt;
    const std::optional<bool> bcast_batch = std::nullopt;
    const MemoryConfig output_mem_config = tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG;
    const std::optional<DataType> output_dtype = std::nullopt;
    const std::optional<DeviceComputeKernelConfig> compute_kernel_config = std::nullopt;
    const bool untilize_out = false;
    const std::optional<const CoreCoord> user_core_coord = std::nullopt;
    const std::optional<UnaryWithParam> user_fused_activation = std::nullopt;
    const bool user_run_batched = false;
    const bool transpose_a = false;
    const bool transpose_b = false;
    const std::optional<const tt::tt_metal::Tile> output_tile;
    const std::optional<const GlobalCircularBuffer> global_cb;
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id;

    void validate(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        const std::vector<std::optional<Tensor>>& optional_output_tensors = {std::nullopt}) const;
    std::vector<ttnn::TensorSpec> compute_output_specs(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<Tensor>>& optional_output_tensors = {std::nullopt},
        const std::vector<std::optional<const Tensor>>& optional_input_tensors = {std::nullopt}) const;
    std::vector<Tensor> create_output_tensors(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<Tensor>>& optional_output_tensors = {std::nullopt}) const;
    tt::tt_metal::operation::CacheableMeshWorkload<std::vector<Tensor>> create_mesh_workload(
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        std::vector<Tensor>& output_tensors) const;
    tt::tt_metal::operation::OpPerformanceModel create_op_performance_model(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        std::vector<Tensor>& output_tensors) const;
};

Matmul create_matmul_struct(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const struct Matmul& parameters,
    const std::vector<std::optional<Tensor>>& optional_output_tensors = {std::nullopt});

struct SparseMatmul {
    uint32_t num_batches;
    const std::optional<const MatmulProgramConfig> program_config = std::nullopt;
    const MemoryConfig output_mem_config = tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG;
    const std::optional<DataType> output_dtype = std::nullopt;
    const std::optional<DeviceComputeKernelConfig> compute_kernel_config = std::nullopt;
    const std::optional<const CoreCoord> user_core_coord = std::nullopt;
    const std::optional<UnaryWithParam> user_fused_activation = std::nullopt;
    const std::optional<const tt::tt_metal::Tile> output_tile;
    const std::optional<const GlobalCircularBuffer> global_cb;
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id;

    void validate(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<Tensor>>& optional_output_tensors = {std::nullopt}) const;
    std::vector<ttnn::TensorSpec> compute_output_specs(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<Tensor>>& optional_output_tensors = {std::nullopt}) const;
    std::vector<Tensor> create_output_tensors(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<Tensor>>& optional_output_tensors = {std::nullopt}) const;
    tt::tt_metal::operation::CacheableMeshWorkload<std::vector<Tensor>> create_mesh_workload(
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const std::vector<Tensor>& input_tensors,
        std::vector<Tensor>& output_tensors) const;
};

SparseMatmul create_sparse_matmul_struct(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const Tensor& sparsity,
    const struct SparseMatmul& parameters,
    const std::vector<std::optional<Tensor>>& optional_output_tensors = {std::nullopt});

matmul_mcast_1d_common_override_variables_t matmul_multi_core_reuse_mcast_1d_optimized_helper(
    tt::tt_metal::Program& program,
    const Tensor& input_tensor_a,
    const std::vector<Tensor>& input_tensors_b,
    const std::optional<const Tensor>& bias,
    const std::vector<Tensor>& output_tensors,
    bool bcast_batch,
    DeviceComputeKernelConfig compute_kernel_config,
    const MatmulProgramConfig& program_config,
    bool untilize_out,
    std::optional<ttnn::experimental::ccl::MatmulFusedOpSignaler>& fused_op_signaler,
    const std::optional<const tt::tt_metal::experimental::GlobalCircularBuffer>& global_cb,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    uint32_t start_cb_index,
    std::optional<CoreRangeSet> restricted_cores);

tt::tt_metal::operation::ProgramWithCallbacks matmul_multi_core_reuse_mcast_1d_optimized_helper(
    tt::tt_metal::Program& program,
    const Tensor& input_tensor_a,
    const std::vector<Tensor>& input_tensors_b,
    const std::optional<const Tensor>& bias,
    const std::vector<Tensor>& output_tensors,
    bool bcast_batch,
    DeviceComputeKernelConfig compute_kernel_config,
    const MatmulProgramConfig& program_config,
    bool untilize_out,
    std::optional<ttnn::experimental::ccl::MatmulFusedOpSignaler>& fused_op_signaler,
    const std::optional<const tt::tt_metal::experimental::GlobalCircularBuffer>& global_cb,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id);
tt::tt_metal::operation::ProgramWithCallbacks matmul_multi_core_reuse_mcast_2d_optimized_helper(
    tt::tt_metal::Program& program,
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const std::optional<const Tensor>& bias,
    Tensor& output_tensor,
    bool bcast_batch,
    DeviceComputeKernelConfig compute_kernel_config,
    const MatmulProgramConfig& program_config,
    bool untilize_out,
    std::optional<ttnn::experimental::ccl::MatmulFusedOpSignaler>& matmul_signal_info);

Tensor matmul(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const std::optional<const Tensor>& bias = std::nullopt,
    const struct Matmul& parameters = Matmul{},
    QueueId queue_id = DefaultQueueId,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt);

std::vector<Tensor> matmul_batched_weights(
    const Tensor& input_tensor_a,
    const std::vector<Tensor>& input_tensors_b,
    const std::optional<const Tensor>& bias = std::nullopt,
    const struct Matmul& parameters = Matmul{},
    QueueId queue_id = DefaultQueueId,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt);

Tensor sparse_matmul(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const Tensor& sparsity,
    const struct SparseMatmul& parameters = SparseMatmul{},
    QueueId queue_id = DefaultQueueId,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt);

}  // namespace matmul

}  // namespace operations

}  // namespace ttnn

namespace bmm_op_utils {

std::tuple<uint32_t, uint32_t> get_matmul_subblock_params(
    uint32_t per_core_M,
    uint32_t per_core_N,
    bool per_core_M_equals_subblock_h_constraint,
    bool per_core_N_equals_subblock_w_constraint,
    bool fp32_dest_acc_en);

}  // namespace bmm_op_utils

namespace reuse_mcast_1d_optimized_helpers {
void override_program_parameters(
    const ttnn::operations::matmul::matmul_mcast_1d_common_override_variables_t& shared_variables,
    const void* operation,
    tt::tt_metal::Program& program,
    const std::vector<tt::tt_metal::Tensor>& input_tensors,
    const std::vector<std::optional<const tt::tt_metal::Tensor>>& optional_input_tensors,
    const std::vector<tt::tt_metal::Tensor>& output_tensors);
}  // namespace reuse_mcast_1d_optimized_helpers

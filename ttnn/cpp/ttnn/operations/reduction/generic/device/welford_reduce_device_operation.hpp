// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <variant>

#include "ttnn/tensor/tensor.hpp"
#include "tt_stl/reflection.hpp"

#include "welford_reduce_device_operation_types.hpp"
#include "ttnn/metal_v2_artifacts.hpp"

namespace ttnn::prim {

struct WelfordReducePlan {
    std::uint32_t W = 0;
    std::uint32_t H = 0;
    std::uint32_t W_padded = 0;
    std::uint32_t H_padded = 0;
    std::uint32_t Wt = 0;
    std::uint32_t Ht = 0;
    std::uint32_t HtWt = 0;
    std::uint32_t NC = 0;
    std::uint32_t tile_height = 0;
    std::uint32_t tile_width = 0;
    std::uint32_t input_tile_size = 0;
    std::uint32_t output_tile_size = 0;
    std::uint32_t num_work_units = 0;
    std::uint32_t num_cores = 0;
    std::uint32_t work_group_1 = 0;
    std::uint32_t work_group_2 = 0;
    std::uint32_t reduce_batch_size = 0;
    std::uint32_t post_mul_scaler_bits = 0;
    tt::DataFormat input_format = tt::DataFormat::Float16_b;
    tt::DataFormat output_format = tt::DataFormat::Float16_b;
    tt::DataFormat scratch_format = tt::DataFormat::Float16_b;
    tt::DataFormat combined_format = tt::DataFormat::Float32;
    CoreRangeSet all_cores;
    CoreRangeSet core_group_1;
    CoreRangeSet core_group_2;
    bool reduce_w = false;
    bool reduce_h = false;
    bool reduce_hw = false;
    bool fp32_dest_acc_en = false;
    bool dst_full_sync_en = false;
    bool is_std = false;
    bool use_post_mul = false;
    bool narrow_scratch_to_bf16 = false;
    bool use_sfpu_leaf_combine = false;
    bool use_l1_replay = false;
};

struct WelfordReduceDeviceOperation {
    using operation_attributes_t = WelfordReduceParams;
    using tensor_args_t = Tensor;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;

    struct WelfordReduceProgramFactory {
        static WelfordReducePlan select_plan(
            const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

        static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    };

    using program_factory_t = std::variant<WelfordReduceProgramFactory>;

    static program_factory_t select_program_factory(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static ttsl::hash::hash_t compute_program_hash(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static void validate_on_program_cache_miss(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static spec_return_value_t compute_output_specs(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);
};

ttnn::Tensor welford_reduce(
    const Tensor& input_tensor,
    tt::tt_metal::ReduceOpMath reduce_math,
    tt::tt_metal::ReduceOpDim reduce_dim,
    float scalar,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const std::optional<tt::tt_metal::DataType>& output_dtype,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    bool correction,
    const std::optional<CoreRangeSet>& sub_core_grids,
    uint32_t reduce_batch_size = 1);

}  // namespace ttnn::prim

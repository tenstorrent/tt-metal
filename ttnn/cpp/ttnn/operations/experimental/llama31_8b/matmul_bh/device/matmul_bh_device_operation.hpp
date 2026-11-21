// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>
#include <vector>

#include "ttnn/device_operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/operations/experimental/llama31_8b/matmul_bh/device/factory/minimal.hpp"
#include "ttnn/operations/experimental/minimal_matmul/device/minimal_matmul_device_operation.hpp"
#include "ttnn/operations/matmul/device/matmul_device_operation_types.hpp"

namespace ttnn::prim::llama31_8b_matmul_bh {

struct Mcast1DProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const MatmulParams& operation_attributes,
        const MatmulInputs& tensor_args,
        std::vector<ttnn::Tensor>& tensor_return_value);
};

struct Mcast2DProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const MatmulParams& operation_attributes,
        const MatmulInputs& tensor_args,
        std::vector<ttnn::Tensor>& tensor_return_value);
};

struct DramShardedProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const MatmulParams& operation_attributes,
        const MatmulInputs& tensor_args,
        std::vector<ttnn::Tensor>& tensor_return_value);
};

struct MatmulBhDeviceOperation {
    using operation_attributes_t = MatmulParams;
    using tensor_args_t = MatmulInputs;
    using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;
    using program_factory_t = std::variant<Mcast1DProgramFactory, Mcast2DProgramFactory, DramShardedProgramFactory>;

    static program_factory_t select_program_factory(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static void validate_on_program_cache_miss(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static spec_return_value_t compute_output_specs(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);
};

bool covers(const MatmulParams& operation_attributes, const MatmulInputs& tensor_args);

bool fits(const MatmulParams& operation_attributes, const MatmulInputs& tensor_args);

}  // namespace ttnn::prim::llama31_8b_matmul_bh

namespace ttnn::experimental::prim::llama31_8b_matmul_bh {

struct MinimalBhDeviceOperation {
    using operation_attributes_t = MinimalMatmulParams;
    using tensor_args_t = MinimalMatmulInputs;
    using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;
    using program_factory_t = std::variant<MinimalProgramFactory>;

    static ttsl::hash::hash_t compute_program_hash(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static void validate_on_program_cache_hit(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static void validate_on_program_cache_miss(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static spec_return_value_t compute_output_specs(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);
};

}  // namespace ttnn::experimental::prim::llama31_8b_matmul_bh

namespace ttnn::experimental::prim::llama31_8b_matmul_bh {

struct QkvHeadsParams {
    MinimalMatmulParams matmul;
    uint32_t num_heads = 0;
    uint32_t num_kv_heads = 0;
};

struct QkvHeadsProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const QkvHeadsParams& operation_attributes,
        const MinimalMatmulInputs& tensor_args,
        std::vector<Tensor>& tensor_return_value);

    static tt::tt_metal::experimental::ProgramRunArgs override_runtime_arguments(
        const QkvHeadsParams& operation_attributes,
        const MinimalMatmulInputs& tensor_args,
        std::vector<Tensor>& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& coord = std::nullopt);
};

struct QkvHeadsDeviceOperation {
    using operation_attributes_t = QkvHeadsParams;
    using tensor_args_t = MinimalMatmulInputs;
    using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;
    using program_factory_t = std::variant<QkvHeadsProgramFactory>;

    static ttsl::hash::hash_t compute_program_hash(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static void validate_on_program_cache_hit(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static void validate_on_program_cache_miss(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static spec_return_value_t compute_output_specs(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);
};

bool minimal_fits(const MinimalMatmulParams& operation_attributes, const MinimalMatmulInputs& tensor_args);

}  // namespace ttnn::experimental::prim::llama31_8b_matmul_bh

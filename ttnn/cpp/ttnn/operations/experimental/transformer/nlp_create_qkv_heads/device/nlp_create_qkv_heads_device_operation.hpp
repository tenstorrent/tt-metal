// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/operation.hpp"
#include <variant>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"              // exposes ttnn::MemoryConfig alias used in member/signature declarations
#include "ttnn/distributed/types.hpp"  // exposes ttnn::MeshCoordinate used in override_runtime_arguments()
#include "ttnn/metal_v2_artifacts.hpp"

#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

namespace ttnn::operations::experimental::transformer {

struct NlpCreateHeadsDeviceOperation {
    struct operation_attributes_t {
        uint32_t num_q_heads;
        uint32_t num_kv_heads;
        uint32_t head_dim;
        bool transpose_k_heads;
        bool kv_tied;
        MemoryConfig output_mem_config;
        std::optional<uint32_t> q_head_split;  // Q-only: output 0/1 hold the two channel regions.
    };

    struct tensor_args_t {
        const Tensor& input_tensor_q;
        const std::optional<Tensor>& input_tensor_kv;
        std::vector<std::optional<Tensor>> optional_output_tensors;
    };

    using spec_return_value_t =
        std::tuple<tt::tt_metal::TensorSpec, tt::tt_metal::TensorSpec, tt::tt_metal::TensorSpec>;
    using tensor_return_value_t = std::tuple<Tensor, Tensor, Tensor>;

    // Both factories are Metal 2.0 spec factories on CustomProgramSpecFactoryConcept: the framework builds
    // and caches the Program from the returned ProgramSpec, and re-applies the ProgramRunArgs returned by
    // override_runtime_arguments on every program-cache hit.
    struct Interleaved {
        static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);

        static tt::tt_metal::experimental::ProgramRunArgs override_runtime_arguments(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value,
            const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
    };

    struct Sharded {
        static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);

        static tt::tt_metal::experimental::ProgramRunArgs override_runtime_arguments(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value,
            const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
    };

    using program_factory_t = std::variant<Interleaved, Sharded>;

    // Mandatory methods

    // Select the program factory based on the operation attributes and tensor args
    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    // Validate the operation when it creates a program. Usually will have more checks
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    // Validate the operation when it reuses a program. Usually will have less checks
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);

    // Compute the output shapes based on the operation attributes and tensor args
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    // Create the output tensors based on the operation attributes and tensor args
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::experimental::transformer

namespace ttnn::prim {
std::tuple<Tensor, Tensor, Tensor> nlp_create_qkv_heads(
    const Tensor& input_tensor_q,
    const std::optional<Tensor>& input_tensor_kv,
    uint32_t num_q_heads,
    std::optional<uint32_t> num_kv_heads,
    uint32_t head_dim,
    bool transpose_k_heads,
    bool kv_tied = false,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<std::vector<std::optional<Tensor>>>& optional_output_tensors = std::nullopt,
    std::optional<uint32_t> q_head_split = std::nullopt);
}  // namespace ttnn::prim

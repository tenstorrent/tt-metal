// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include "ttnn/operation.hpp"
#include <variant>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"  // exposes ttnn::MemoryConfig alias used in member/signature declarations
#include "ttnn/distributed/types.hpp"  // exposes ttnn::MeshCoordinate used in get_dynamic_runtime_args()

#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/experimental/program_descriptor_patching.hpp>

namespace ttnn::operations::experimental::transformer {

struct NlpCreateHeadsBoltzDeviceOperation {
    struct operation_attributes_t {
        uint32_t num_q_heads;
        uint32_t num_kv_heads;
        uint32_t head_dim;
        bool transpose_k_heads;
        MemoryConfig output_mem_config;
    };

    struct tensor_args_t {
        const Tensor& input_tensor_q;
        const std::optional<Tensor>& input_tensor_kv;
        std::vector<std::optional<Tensor>> optional_output_tensors;
    };

    using spec_return_value_t = std::tuple<ttnn::TensorSpec, ttnn::TensorSpec, ttnn::TensorSpec>;
    using tensor_return_value_t = std::tuple<Tensor, Tensor, Tensor>;

    struct Interleaved {
        static tt::tt_metal::ProgramDescriptor create_descriptor(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    };

    struct Sharded {
        static tt::tt_metal::ProgramDescriptor create_descriptor(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
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

    // Re-apply the Sharded factory's computed input-buffer addresses on every cache hit.
    // The Sharded reader/writer bake raw base addresses AND per-core `base + head_offset` start
    // addresses as uint32 runtime args; a plain Buffer* binding can only express the bare base, so
    // these address-derived slots are refreshed here instead (the output CBs are patched via their
    // `.buffer` bindings).  Defined in nlp_create_qkv_heads_boltz_program_factory.cpp so it can reuse
    // the shared per-core arg builder that create_descriptor() uses. Interleaved returns no dynamic
    // args.
    static std::vector<tt::tt_metal::DynamicRuntimeArg> get_dynamic_runtime_args(
        const operation_attributes_t&,
        const tensor_args_t&,
        tensor_return_value_t&,
        const std::optional<ttnn::MeshCoordinate>& = std::nullopt);
};

}  // namespace ttnn::operations::experimental::transformer

namespace ttnn::prim {
std::tuple<Tensor, Tensor, Tensor> nlp_create_qkv_heads_boltz(
    const Tensor& input_tensor_q,
    const std::optional<Tensor>& input_tensor_kv,
    uint32_t num_q_heads,
    std::optional<uint32_t> num_kv_heads,
    uint32_t head_dim,
    bool transpose_k_heads,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<std::vector<std::optional<Tensor>>>& optional_output_tensors);
}  // namespace ttnn::prim

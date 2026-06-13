// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>
#include "ttnn/operation.hpp"
#include <variant>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"  // exposes ttnn::MemoryConfig alias used in member/signature declarations
#include "ttnn/distributed/types.hpp"

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

    // Per-core runtime args for the Sharded factory, derived purely from
    // (operation_attributes, tensor_args, tensor_return_value). This is the SINGLE SOURCE OF TRUTH
    // for both Sharded::create_descriptor() (cache miss) and get_dynamic_runtime_args() (cache-hit
    // re-apply). The reader/writer arg vectors are the complete per-core vectors create_descriptor()
    // emplaces; addr_indices records the slots that hold raw buffer-derived addresses (q_base,
    // q_start, k/v_base, k/v_start), which are the only slots that change across dispatches and so
    // are re-applied dynamically on every cache hit.
    struct ShardedPerCoreArgs {
        std::vector<CoreCoord> cores;
        std::vector<tt::tt_metal::KernelDescriptor::CoreRuntimeArgs> reader_args;
        std::vector<tt::tt_metal::KernelDescriptor::CoreRuntimeArgs> writer_args;
        // Address-derived rt-arg indices, identical for the reader and writer arg vectors:
        //   [6]=q_base_addr, [7]=q_start_addr, [15]=k/v_base_addr, [16]=k/v_start_addr.
        std::vector<uint32_t> addr_indices;
    };

    static ShardedPerCoreArgs compute_sharded_per_core_args(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);

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

    // Re-apply the address-derived runtime args on every cache hit. The Sharded factory bakes raw
    // q/k/v base + per-core start addresses into the reader/writer rt-args; these change whenever the
    // input (or optional kv) buffer is re-allocated, so they must be recomputed and re-applied.
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

// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
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

struct NlpCreateHeadsDeviceOperation;

// Per-core runtime args for the Sharded factory, derived once and consumed by BOTH
// Sharded::create_descriptor() (cache-miss build) and get_dynamic_runtime_args() (cache-hit re-apply).
// This is the SINGLE SOURCE OF TRUTH for the per-core address state machine: the reader/writer rt-arg
// vectors are identical except for the slots the factory mutates between the two emplaces. The address
// slots (q_base/q_start/k_base|v_base/k_start|v_start) are the ones that change across dispatches when the
// input (or optional kv) buffer is re-allocated; their positions are recorded in addr_indices so
// get_dynamic_runtime_args() can re-apply exactly those.
struct NlpCreateQkvHeadsShardedPerCoreArgs {
    std::vector<tt::tt_metal::CoreCoord> cores;
    // Indexed by position in `cores`; full reader/writer rt-arg vectors for each work core.
    std::vector<std::vector<uint32_t>> reader_args;
    std::vector<std::vector<uint32_t>> writer_args;
    std::vector<bool> is_work_core;  // every core in this op's loop is a work core (kept for symmetry)
    // Buffer-address arg positions, identical for the reader and writer arg vectors:
    //   [6]=q_base_addr, [7]=q_start_addr, [15]=k_base_addr/v_base_addr, [16]=k_start_addr/v_start_addr.
    std::vector<uint32_t> addr_indices;
};

struct NlpCreateHeadsDeviceOperation {
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

    // Re-apply the address-derived runtime args on every cache hit. The Sharded factory bakes raw
    // q/k/v base + per-core start addresses into the reader/writer rt-args; these change whenever the
    // input (or optional kv) buffer is re-allocated, so they must be recomputed and re-applied.
    static std::vector<tt::tt_metal::DynamicRuntimeArg> get_dynamic_runtime_args(
        const operation_attributes_t&,
        const tensor_args_t&,
        tensor_return_value_t&,
        const std::optional<ttnn::MeshCoordinate>& = std::nullopt);
};

// Derive the Sharded per-core reader/writer rt-args once (single source of truth, see struct doc above).
NlpCreateQkvHeadsShardedPerCoreArgs compute_nlp_create_qkv_heads_sharded_per_core_args(
    const NlpCreateHeadsDeviceOperation::operation_attributes_t& operation_attributes,
    const NlpCreateHeadsDeviceOperation::tensor_args_t& tensor_args,
    NlpCreateHeadsDeviceOperation::tensor_return_value_t& tensor_return_value);

}  // namespace ttnn::operations::experimental::transformer

namespace ttnn::prim {
std::tuple<Tensor, Tensor, Tensor> nlp_create_qkv_heads(
    const Tensor& input_tensor_q,
    const std::optional<Tensor>& input_tensor_kv,
    uint32_t num_q_heads,
    std::optional<uint32_t> num_kv_heads,
    uint32_t head_dim,
    bool transpose_k_heads,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<std::vector<std::optional<Tensor>>>& optional_output_tensors);
}  // namespace ttnn::prim

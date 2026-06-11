// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/operation.hpp"
#include <variant>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"  // exposes ttnn::MemoryConfig alias used in member/signature declarations

#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/metal2_artifacts.hpp"
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

namespace ttnn::operations::experimental::transformer {

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

    // Metal 2.0 factories. Both build a ProgramSpec + ProgramRunArgs (ProgramArtifacts) in place of the
    // legacy ProgramDescriptor.
    //
    // Interleaved — degenerate ProgramSpecFactoryConcept. All Q/K/V inputs and outputs flow through clean
    // TensorAccessor bindings (TensorParameter + TensorBinding), so the cache-hit path's UpdateTensorArgs
    // refreshes every address; all run-args live in create_program_artifacts' run_params. (The optional
    // kv input gets a TensorParameter/binding only when present — that's structural, so it's fine for it
    // to differ across cache entries.)
    struct Interleaved {
        static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    };

    // Sharded — degenerate concept WITH the per-enqueue split ("++"). The sharded reader/writer address
    // the input via raw (noc_x, noc_y, addr) endpoints rather than TensorAccessor bindings, so the source
    // addresses are baked as plain runtime args. Those addresses (q/k/v base + per-core start offsets)
    // can change between two dispatches that share a cache entry, so they live in create_per_enqueue_args
    // and are re-applied on every dispatch via UpdateProgramRunArgs. The output tensors back the Q/K/V
    // DFBs via borrowed_from, so they are declared as TensorParameters and refreshed the same way.
    struct Sharded {
        static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
        static tt::tt_metal::experimental::ProgramRunArgs create_per_enqueue_args(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value,
            const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate);
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
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<std::vector<std::optional<Tensor>>>& optional_output_tensors);
}  // namespace ttnn::prim

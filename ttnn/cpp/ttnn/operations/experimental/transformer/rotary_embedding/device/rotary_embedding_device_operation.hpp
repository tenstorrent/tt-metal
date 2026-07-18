// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/experimental/transformer/rotary_embedding/device/rotary_embedding_device_operation_types.hpp"
#include "ttnn/operations/experimental/transformer/rotary_embedding/device/rotary_embedding_program_factory.hpp"
#include "ttnn/types.hpp"
#include <tt-metalium/experimental/program_descriptor_patching.hpp>

namespace ttnn::experimental::prim {

struct RotaryEmbeddingDeviceOperation {
    using operation_attributes_t = RotaryEmbeddingParams;
    using tensor_args_t = RotaryEmbeddingInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<RotaryEmbeddingProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t& args, const tensor_args_t&);

    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);

    // Decode mode (token_idx set) derives cos_sin_start_id / cos_sin_offset from token_idx and bakes
    // them into static reader/writer runtime args, while token_idx is deliberately excluded from
    // compute_program_hash so successive decode positions cache-hit the same program.  Those two
    // scalars must therefore be re-applied on every cache hit -- otherwise the cached program keeps
    // the first token's offsets and every later token reads the wrong cos/sin rows.  override_runtime_arguments
    // re-derives the descriptor and re-applies them (prefill derives offsets from hashed shapes -- a no-op there).
    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& output,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
};

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {
Tensor rotary_embedding(
    const Tensor& input,
    const Tensor& cos,
    const Tensor& sin,
    uint32_t seq_len,
    std::optional<uint32_t> token_idx,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    ttnn::DeviceComputeKernelConfig compute_kernel_config);
}  // namespace ttnn::prim

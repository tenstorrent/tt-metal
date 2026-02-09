// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "nlp_create_qkv_heads_decode_device_operation_types.hpp"
#include "nlp_create_qkv_heads_decode_interleaved_program_factory.hpp"
#include "nlp_create_qkv_heads_decode_sharded_program_factory.hpp"
#include "nlp_create_qkv_heads_decode_sharded_subcoregrid_program_factory.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/decorators.hpp"

namespace ttnn::operations::experimental::nlp_create_qkv_heads_decode {

struct NLPCreateQKVHeadsDecodeDeviceOperation {
    using operation_attributes_t = NlpCreateQkvHeadsDecodeParams;
    using tensor_args_t = NlpCreateQkvHeadsDecodeInputs;
    using spec_return_value_t = nlp_create_qkv_heads_decode::spec_return_value_t;
    using tensor_return_value_t = nlp_create_qkv_heads_decode::tensor_return_value_t;
    using program_factory_t = std::variant<
        program::NLPCreateQKVHeadsDecodeInterleavedProgramFactory,
        program::NLPCreateQKVHeadsDecodeShardedProgramFactory,
        program::NLPCreateQKVHeadsDecodeShardedSubcoregridProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t&);
};

}  // namespace ttnn::operations::experimental::nlp_create_qkv_heads_decode

namespace ttnn::prim {
std::vector<Tensor> nlp_create_qkv_heads_decode(
    const Tensor& input_tensor,
    uint32_t num_q_heads,
    uint32_t num_kv_heads,
    uint32_t head_dim,
    bool overlap_qk_coregrid,
    bool input_on_subcoregrids,
    const std::optional<const Tensor>& batch_offset,
    std::optional<uint32_t> slice_size,
    const tt::tt_metal::MemoryConfig& output_mem_config);
}  // namespace ttnn::prim

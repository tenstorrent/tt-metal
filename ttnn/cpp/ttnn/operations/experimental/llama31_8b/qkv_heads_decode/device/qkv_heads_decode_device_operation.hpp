// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>
#include <vector>

#include "ttnn/device_operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/operations/experimental/transformer/nlp_create_qkv_heads_decode/device/nlp_create_qkv_heads_decode_device_operation_types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim::llama31_8b_qkv_heads_decode {

struct QkvHeadsDecodeProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const ttnn::experimental::prim::NlpCreateQkvHeadsDecodeParams& operation_attributes,
        const ttnn::experimental::prim::NlpCreateQkvHeadsDecodeInputs& tensor_args,
        std::vector<Tensor>& output);
};

struct QkvHeadsDecodeDeviceOperation {
    using operation_attributes_t = ttnn::experimental::prim::NlpCreateQkvHeadsDecodeParams;
    using tensor_args_t = ttnn::experimental::prim::NlpCreateQkvHeadsDecodeInputs;
    using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;
    using program_factory_t = std::variant<QkvHeadsDecodeProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::prim::llama31_8b_qkv_heads_decode

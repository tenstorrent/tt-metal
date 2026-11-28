// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>

#include "ttnn/device_operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim::llama31_8b_concat_heads_decode {

struct ConcatHeadsDecodeParams {
    uint32_t num_heads = 0;
};

struct ConcatHeadsDecodeInputs {
    Tensor input;
};

struct ConcatHeadsDecodeProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const ConcatHeadsDecodeParams& operation_attributes,
        const ConcatHeadsDecodeInputs& tensor_args,
        Tensor& output);
};

struct ConcatHeadsDecodeDeviceOperation {
    using operation_attributes_t = ConcatHeadsDecodeParams;
    using tensor_args_t = ConcatHeadsDecodeInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<ConcatHeadsDecodeProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::prim::llama31_8b_concat_heads_decode

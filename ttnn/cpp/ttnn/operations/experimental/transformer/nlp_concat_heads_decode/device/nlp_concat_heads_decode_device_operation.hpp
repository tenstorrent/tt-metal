// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/decorators.hpp"
#include "nlp_concat_heads_decode_device_operation_types.hpp"
#include "nlp_concat_heads_decode_program_factory.hpp"
#include "nlp_concat_heads_decode_subcoregrids_program_factory.hpp"

namespace ttnn::experimental::prim {

struct NLPConcatHeadsDecodeDeviceOperation {
    using operation_attributes_t = NlpConcatHeadsDecodeParams;
    using tensor_args_t = NlpConcatHeadsDecodeInputs;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t =
        std::variant<NLPConcatHeadsDecodeProgramFactory, NLPConcatHeadsDecodeSubcoregridsProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t& args, const tensor_args_t&);
};

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {
Tensor nlp_concat_heads_decode(
    const Tensor& input_tensor,
    uint32_t num_heads,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& preallocated_output = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
}  // namespace ttnn::prim

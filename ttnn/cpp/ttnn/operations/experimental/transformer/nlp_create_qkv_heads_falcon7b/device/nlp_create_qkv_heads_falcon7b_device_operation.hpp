// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"
#include "nlp_create_qkv_heads_falcon7b_device_operation_types.hpp"

namespace ttnn::experimental::prim {

struct NlpCreateHeadsFalcon7BDeviceOperation {
    using operation_attributes_t = NlpCreateQkvHeadsFalcon7bParams;
    using tensor_args_t = Tensor;
    using spec_return_value_t = NlpCreateQkvHeadsFalcon7bResultSpec;
    using tensor_return_value_t = NlpCreateQkvHeadsFalcon7bResult;

    struct NlpCreateQkvHeadsFalcon7BProgramFactory {
        // The only per-dispatch state is the four buffer addresses (input, q, k, v); every other
        // per-core arg is derived from the input's padded shape and the compute grid, both part of the
        // program hash. The tensor bindings are therefore the whole cache-hit refresh, so there is no
        // override.
        static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    };
    using program_factory_t = std::variant<NlpCreateQkvHeadsFalcon7BProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t&);
};

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {
ttnn::experimental::prim::NlpCreateQkvHeadsFalcon7bResult nlp_create_qkv_heads_falcon7b(
    const Tensor& input, const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt);
}  // namespace ttnn::prim

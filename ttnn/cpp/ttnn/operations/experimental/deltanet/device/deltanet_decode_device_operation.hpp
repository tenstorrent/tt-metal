// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "deltanet_decode_program_factory.hpp"
#include "deltanet_decode_device_operation_types.hpp"

namespace ttnn::operations::experimental::deltanet {

struct DeltaNetDecodeDeviceOperation {
    using operation_attributes_t = DeltaNetDecodeParams;
    using tensor_args_t = DeltaNetDecodeInputs;

    using spec_return_value_t = std::vector<ttnn::TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;

    using program_factory_t = std::variant<DeltaNetDecodeProgramFactory>;

    static program_factory_t select_program_factory(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static void validate_on_program_cache_miss(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static void validate_on_program_cache_hit(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static spec_return_value_t compute_output_specs(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args);
};

}  // namespace ttnn::operations::experimental::deltanet

namespace ttnn::prim {

std::vector<Tensor> deltanet_decode(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const Tensor& decay,
    const Tensor& beta,
    const Tensor& state,
    uint32_t num_heads,
    uint32_t k_head_dim,
    uint32_t v_head_dim,
    const std::optional<tt::tt_metal::MemoryConfig>& output_memory_config = std::nullopt);

}  // namespace ttnn::prim

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"

#include <tt-metalium/program_descriptors.hpp>

#include <memory>
#include <optional>
#include <tuple>

namespace ttnn::operations::qr {

struct QrDeviceOperation {
    struct operation_attributes_t {
        const MemoryConfig memory_config;

        static constexpr auto attribute_names = std::forward_as_tuple("memory_config");
        auto attribute_values() const { return std::forward_as_tuple(memory_config); }
    };

    struct tensor_args_t {
        const Tensor& input;
    };

    using spec_return_value_t = std::tuple<tt::tt_metal::TensorSpec, tt::tt_metal::TensorSpec>;
    using tensor_return_value_t = std::tuple<Tensor, Tensor>;

    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::qr

namespace ttnn::prim {
std::tuple<Tensor, Tensor> qr(
    const Tensor& input, const std::optional<MemoryConfig>& memory_config = std::nullopt);
}  // namespace ttnn::prim

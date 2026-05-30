// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include <tt-metalium/base_types.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/tensor/memory_config/memory_config.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"
#include "prefix_scan_program_factory.hpp"

#include "prefix_scan_device_operation_types.hpp"

namespace ttnn::experimental::prim {

struct PrefixScanDeviceOperation {
    using operation_attributes_t = PrefixScanParams;
    using tensor_args_t = PrefixScanInputs;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<PrefixScanProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t&);
};

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

ttnn::experimental::prim::PrefixScanDeviceOperation::tensor_return_value_t prefix_scan(
    const Tensor& a,
    const Tensor& bx,
    const Tensor& h_prev,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    std::optional<tt::tt_metal::DataType> dtype = std::nullopt,
    std::optional<tt::tt_metal::MathFidelity> math_fidelity = std::nullopt);

}  // namespace ttnn::prim

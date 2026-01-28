// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "prefix_scan_program_factory.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/decorators.hpp"

#include "prefix_scan_device_operation_types.hpp"

namespace ttnn::experimental::prim {

struct PrefixScanDeviceOperation {
    using operation_attributes_t = PrefixScanParams;
    using tensor_args_t = PrefixScanInputs;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<PrefixScanProgramFactory>;
    using shared_variables_t = PrefixScanProgramFactory::shared_variables_t;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t&);

    static tt::stl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

ttnn::experimental::prim::PrefixScanDeviceOperation::tensor_return_value_t prefix_scan(
    const Tensor& a,
    const Tensor& bx,
    const Tensor& h_prev,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    std::optional<DataType> dtype = std::nullopt,
    std::optional<MathFidelity> math_fidelity = std::nullopt);

}  // namespace ttnn::prim

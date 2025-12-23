// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ema_device_operation_types.hpp"
#include "ema_program_factory.hpp"
#include "ttnn/decorators.hpp"

#include <optional>
#include <variant>

namespace ttnn::operations::reduction::ema {

struct EmaDeviceOperation {
    using operation_attributes_t = ema::operation_attributes_t;
    using tensor_args_t = ema::tensor_args_t;
    using spec_return_value_t = ema::spec_return_value_t;
    using tensor_return_value_t = ema::tensor_return_value_t;
    using program_factory_t = std::variant<program::EmaProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& input,
        float alpha,
        CoreCoord grid_size,
        const tt::tt_metal::MemoryConfig& output_mem_config,
        const DeviceComputeKernelConfig& compute_kernel_config,
        std::optional<Tensor> optional_output_tensor);
};

}  // namespace ttnn::operations::reduction::ema

namespace ttnn::prim {
constexpr auto ema_device =
    ttnn::register_operation<"ttnn::prim::ema_device", ttnn::operations::reduction::ema::EmaDeviceOperation>();
}  // namespace ttnn::prim

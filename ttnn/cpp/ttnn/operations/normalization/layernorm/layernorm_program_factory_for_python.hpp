// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <optional>
#include <variant>

#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/normalization/layernorm/device/layernorm_device_operation_types.hpp"

namespace ttnn::for_python {

// Metal 1.0 copies of the LayerNorm program factories, frozen for the Python fusion framework.
// Not part of LayerNormDeviceOperation and not on any dispatch path — do not port them to Metal 2.0.
struct LayerNormMultiCoreProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const ttnn::prim::LayerNormParams& operation_attributes,
        const ttnn::prim::LayerNormInputs& tensor_args,
        Tensor& tensor_return_value,
        const std::optional<CoreRangeSet>& core_range_set = std::nullopt);

    static CoreRangeSet default_core_range(tt::tt_metal::IDevice* device);
};

struct LayerNormShardedProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const ttnn::prim::LayerNormParams& operation_attributes,
        const ttnn::prim::LayerNormInputs& tensor_args,
        Tensor& tensor_return_value,
        const std::optional<CoreRangeSet>& core_range_set = std::nullopt);
};

using LayerNormProgramFactory = std::variant<LayerNormMultiCoreProgramFactory, LayerNormShardedProgramFactory>;

LayerNormProgramFactory select_layernorm_program_factory(
    const ttnn::prim::LayerNormParams& operation_attributes, const ttnn::prim::LayerNormInputs& tensor_args);

}  // namespace ttnn::for_python

// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "intimg_device_operation_types.hpp"

#include <cstdint>

#include <optional>
#include <type_traits>
#include <variant>

#include "hostdevcommon/kernel_structs.h"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"

namespace ttnn::experimental::prim {

using namespace tt::tt_metal;
using namespace ttsl;

struct IntImgDeviceOperation {
    using operation_attributes_t = IntImgParams;
    using tensor_args_t = Tensor;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;

    using invocation_result_t = std::tuple<operation_attributes_t, tensor_args_t>;
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    // Metal 2.0 factory (MetalV2FactoryConcept). Emits a ProgramSpec + ProgramRunArgs from the single
    // interleaved config. Wrapped in a single-alternative program_factory_t so the framework's
    // Metal 2.0 adapter selects it (the DirectDescriptor fallback wraps create_descriptor, not this).
    struct ProgramFactory {
        static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    };

    using program_factory_t = std::variant<ProgramFactory>;
};

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

Tensor intimg(const Tensor& input_tensor);

}  // namespace ttnn::prim

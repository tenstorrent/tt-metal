// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include "ttnn/operations/wavelet/device/ilwt_1d_program_factory.hpp"

namespace ttnn::prim {

struct Ilwt1DDeviceOperation {
    using operation_attributes_t = Ilwt1DParams;
    using tensor_args_t = Ilwt1DInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<Ilwt1DProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

Tensor ilwt(
    const Tensor& approximation,
    const Tensor& detail,
    operations::wavelet::SchemeId scheme_id,
    operations::wavelet::BoundaryMode boundary_mode,
    uint32_t original_length,
    const MemoryConfig& output_memory_config,
    const std::optional<Tensor>& preallocated_output = std::nullopt);

}  // namespace ttnn::prim

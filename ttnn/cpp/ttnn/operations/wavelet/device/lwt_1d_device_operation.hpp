// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include "ttnn/operations/wavelet/device/lwt_1d_program_factory.hpp"

namespace ttnn::prim {

struct Lwt1DDeviceOperation {
    using operation_attributes_t = Lwt1DParams;
    using tensor_args_t = Lwt1DInputs;
    using spec_return_value_t = Lwt1DOutputSpecs;
    using tensor_return_value_t = Lwt1DOutputs;
    using program_factory_t = std::variant<Lwt1DProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

Lwt1DOutputs lwt(
    const Tensor& input,
    operations::wavelet::SchemeId scheme_id,
    operations::wavelet::BoundaryMode boundary_mode,
    const MemoryConfig& output_memory_config,
    const std::optional<Lwt1DOutputs>& preallocated_outputs = std::nullopt);

}  // namespace ttnn::prim

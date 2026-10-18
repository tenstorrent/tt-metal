// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include <tt-metalium/program.hpp>
#include <tt-metalium/program_descriptors.hpp>

#include "generalized_moe_gate_device_operation_types.hpp"
#include "ttnn/distributed/types.hpp"

namespace ttnn::operations::experimental::deepseek::moe::generalized_moe_gate::program {

struct GeneralizedMoeGateProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);

    // Tensor-backed circular-buffer bases. Compile-time args stay in the default program hash.
    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& coord = std::nullopt);
};

}  // namespace ttnn::operations::experimental::deepseek::moe::generalized_moe_gate::program

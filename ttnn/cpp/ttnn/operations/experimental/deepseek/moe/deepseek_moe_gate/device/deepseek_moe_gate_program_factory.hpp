// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

#include "deepseek_moe_gate_device_operation_types.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/metal_v2_artifacts.hpp"

namespace ttnn::operations::experimental::deepseek::moe::deepseek_moe_gate::program {

struct DeepseekMoeGateProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);

    // Tensor-backed dataflow-buffer bases. Compile-time args stay in the default program hash.
    static tt::tt_metal::experimental::ProgramRunArgs override_runtime_arguments(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& coord = std::nullopt);
};

}  // namespace ttnn::operations::experimental::deepseek::moe::deepseek_moe_gate::program

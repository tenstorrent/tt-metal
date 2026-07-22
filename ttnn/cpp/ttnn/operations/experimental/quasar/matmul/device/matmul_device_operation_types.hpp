// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <tuple>

#include "ttnn/operations/experimental/quasar/matmul/device/config/matmul_program_config_types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "tt-metalium/global_circular_buffer.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operation.hpp"  // for DEFAULT_OUTPUT_MEMORY_CONFIG

namespace ttnn::prim::qsr {

struct MatmulParams {
    std::optional<operations::experimental::quasar::matmul::MatmulProgramConfig> program_config = std::nullopt;
    std::optional<bool> bcast_batch = std::nullopt;
    tt::tt_metal::MemoryConfig output_mem_config = tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG;
    std::optional<tt::tt_metal::DataType> output_dtype = std::nullopt;
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt;
    bool untilize_out = false;
    std::optional<CoreCoord> user_core_coord = std::nullopt;
    std::optional<ttnn::operations::unary::UnaryWithParam> user_fused_activation = std::nullopt;
    bool user_run_batched = false;
    bool transpose_a = false;
    bool transpose_b = false;
    std::optional<tt::tt_metal::Tile> output_tile = std::nullopt;
    std::optional<tt::tt_metal::experimental::GlobalCircularBuffer> global_cb = std::nullopt;
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id = std::nullopt;

    // Compile-time reflection used by the program-cache key (both the 64-bit hash and the
    // collision-free canonical key). Mirrors the Quasar Conv2dDeviceOperation migration
    // (#45821): rather than a custom compute_program_hash, expose the program-selecting
    // attributes so the op stays on the DEFAULT reflection-hash + canonical-key path.
    //
    // A custom compute_program_hash would return only a 64-bit value and, per
    // mesh_device_operation_adapter.hpp::compute_mesh_workload_canonical_key, OPT THE OP OUT of
    // attribute-level canonical-key collision resolution (the canonical key degrades to the
    // op-identity prefix) -- re-exposing exactly the #45821 hash-fold collision. Listing the
    // attributes here instead keeps the exact, collision-free canonical key.
    //
    // Every program-affecting attribute is listed. user_fused_activation (the RELU that
    // distinguishes pure vs bias_relu) is here; program_config (a variant) is encoded by its
    // active index + alternative. Bias PRESENCE lives in tensor_args (MatmulInputs), not here --
    // it is keyed by the same canonical-key traversal of tensor_args (see MatmulInputs).
    static constexpr auto attribute_names = std::forward_as_tuple(
        "program_config",
        "bcast_batch",
        "output_mem_config",
        "output_dtype",
        "compute_kernel_config",
        "untilize_out",
        "user_core_coord",
        "user_fused_activation",
        "user_run_batched",
        "transpose_a",
        "transpose_b",
        "output_tile",
        "global_cb",
        "sub_device_id");
    auto attribute_values() const {
        return std::make_tuple(
            std::cref(this->program_config),
            std::cref(this->bcast_batch),
            std::cref(this->output_mem_config),
            std::cref(this->output_dtype),
            std::cref(this->compute_kernel_config),
            this->untilize_out,
            std::cref(this->user_core_coord),
            std::cref(this->user_fused_activation),
            this->user_run_batched,
            this->transpose_a,
            this->transpose_b,
            std::cref(this->output_tile),
            std::cref(this->global_cb),
            std::cref(this->sub_device_id));
    }
};

struct MatmulInputs {
    std::vector<Tensor> input_tensors;                                // a,b, weights
    std::vector<std::optional<const Tensor>> optional_input_tensors;  // bias
    std::vector<std::optional<Tensor>> optional_output_tensors;       // output
};

}  // namespace ttnn::prim::qsr

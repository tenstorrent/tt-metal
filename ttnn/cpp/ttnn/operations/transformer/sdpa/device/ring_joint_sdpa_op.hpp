// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/transformer/sdpa_config.hpp"
#include "ttnn/operations/experimental/ccl/ring_attention_all_gather_async/device/ring_attention_all_gather_async_op.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::transformer {

struct RingJointScaledDotProductAttention {
    const std::string joint_strategy;
    const std::optional<float> scale;
    const std::size_t logical_n;
    const std::size_t ring_size;

    const tt::tt_metal::MemoryConfig output_mem_config;
    const std::optional<SDPAProgramConfig> program_config;
    const DeviceComputeKernelConfig compute_kernel_config;

    ttnn::RingAttentionAllGatherAsync all_gather_struct;
    const CoreCoord ccl_core_grid_offset;

    void validate(const std::vector<Tensor>& input_tensors) const;

    std::vector<TensorSpec> compute_output_specs(const std::vector<Tensor>& input_tensors) const;

    tt::tt_metal::operation::MeshWorkloadWithCallbacks create_mesh_workload(
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const std::vector<Tensor>& input_tensors,
        std::vector<Tensor>& output_tensors) const;
    tt::tt_metal::operation::ProgramWithCallbacks create_program_at(
        const ttnn::MeshCoordinate& coord,
        const std::vector<Tensor>& input_tensors,
        std::vector<Tensor>& output_tensors) const;

    tt::tt_metal::operation::Hash compute_program_hash(const std::vector<Tensor>& input_tensors) const;

    std::uint32_t get_q_chunk_size() const;
    std::uint32_t get_k_chunk_size() const;

    static constexpr auto attribute_names = std::forward_as_tuple(
        "joint_strategy",
        "scale",
        "logical_n",
        "ring_size",
        "output_mem_config",
        "program_config",
        "compute_kernel_config",
        "ccl_core_grid_offset");
    auto attribute_values() const {
        return std::forward_as_tuple(
            this->joint_strategy,
            this->scale,
            this->logical_n,
            this->ring_size,
            this->output_mem_config,
            this->program_config,
            this->compute_kernel_config,
            this->ccl_core_grid_offset);
    }
};

}  // namespace ttnn::operations::transformer

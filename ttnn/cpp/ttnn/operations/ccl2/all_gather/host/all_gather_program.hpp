// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file all_gather_program.hpp
 * @brief This file contains the host-side code for all_gather operation.
 *
 * The host program is defined using infrastructure provided by:
 * ttnn/api/ttnn/operation.hpp
 */

#pragma once

#include <optional>
#include <utility>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/hal.hpp>
#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/global_semaphore.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include <tt-metalium/fabric.hpp>
#include "ttnn/operations/functions.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/global_semaphore.hpp"

#include "ttnn/tensor/tensor_utils.hpp"

#include "ttnn/operations/ccl2/ccl2_common.hpp"

namespace ttnn {

struct AllGather {
    // Op args
    const std::string arch_name;
    const int32_t dim;
    const ttnn::ccl2::Topology topology;
    const std::optional<ttnn::MemoryConfig>& output_memory_config;
    const std::optional<tt::tt_metal::SubDeviceId> subdevice_id;

    AllGather(
        const int32_t dim,
        const ttnn::ccl2::Topology topology,
        const std::optional<ttnn::MemoryConfig>& output_memory_config,
        const std::optional<tt::tt_metal::SubDeviceId>& subdevice_id) :
        arch_name(tt::tt_metal::hal::get_arch_name()),
        dim(dim),
        topology(topology),
        output_memory_config(output_memory_config),
        subdevice_id(subdevice_id) {}

    // Add attributes method for reflection
    auto attributes() const {
        using tt::stl::reflection::Attribute;
        std::vector<std::tuple<std::string, Attribute>> attrs;

        attrs.emplace_back("arch_name", arch_name);
        attrs.emplace_back("dim", dim);
        attrs.emplace_back("topology", topology);
        attrs.emplace_back("output_memory_config", output_memory_config);
        attrs.emplace_back("subdevice_id", subdevice_id);
        return attrs;
    }

    void validate_with_output_tensors(
        const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const;
    std::vector<ttnn::TensorSpec> compute_output_specs(const std::vector<Tensor>& input_tensors) const;
    tt::tt_metal::operation::MeshWorkloadWithCallbacks create_mesh_workload(
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const std::vector<Tensor>& input_tensors,
        std::vector<Tensor>& output_tensors) const;
    tt::tt_metal::operation::ProgramWithCallbacks create_program_at(
        const ttnn::MeshCoordinate& coord,
        const std::vector<Tensor>& input_tensors,
        std::vector<Tensor>& output_tensors) const;
    tt::tt_metal::operation::Hash compute_program_hash(const std::vector<Tensor>& input_tensors) const;
};

}  // namespace ttnn

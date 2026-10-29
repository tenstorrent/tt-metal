// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/workload_descriptor.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/data_movement/reshape_view/codegen/reshape_codegen_plan.hpp"

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::prim {

struct ReshapeCodegenParams {
    ttnn::Shape logical_output_shape;
    tt::tt_metal::MemoryConfig output_mem_config;
    std::optional<tt::tt_metal::CoreRangeSet> sub_core_grids;
    // Not an attribute: it follows from the shapes, which are. The plan live L1 sizes it to, and the TILE route, are
    // keyed by ReshapeCodegenDeviceOperation::compute_program_hash.
    ttnn::operations::data_movement::reshape_codegen::LaunchPlan plan;

    static constexpr auto attribute_names =
        std::forward_as_tuple("logical_output_shape", "output_mem_config", "sub_core_grids");
    auto attribute_values() const {
        return std::forward_as_tuple(logical_output_shape, output_mem_config, sub_core_grids);
    }
};

struct ReshapeCodegenInputs {
    Tensor input;
};

// Tile-row permutation TILE program. Its plan table is a device tensor the program reads, so the factory builds the
// whole workload and parks the table on it; the table then lives exactly as long as the cached workload.
struct ReshapeCodegenTileRowsProgramFactory {
    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const ReshapeCodegenParams& operation_attributes,
        const ReshapeCodegenInputs& tensor_args,
        Tensor& tensor_return_value,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);
};

// Segment-map TILE program, for widths that are not multiples of 32. Its map is a device tensor parked on the
// workload as the tile-rows table is.
struct ReshapeCodegenTileMappedProgramFactory {
    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const ReshapeCodegenParams& operation_attributes,
        const ReshapeCodegenInputs& tensor_args,
        Tensor& tensor_return_value,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);
};

}  // namespace ttnn::prim

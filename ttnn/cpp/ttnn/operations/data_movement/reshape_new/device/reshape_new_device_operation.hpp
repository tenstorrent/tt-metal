// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <variant>
#include <vector>

#include <tt-metalium/experimental/distributed_tensor/topology/tensor_topology.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/operations/data_movement/reshape_new/device/reshape_new_device_operation_types.hpp"
#include "ttnn/types.hpp"

namespace ttnn::prim {

// Contiguous ranges of work units over the cores of `sub_core_grid`, or of the device's compute grid. Core i takes
// units [unit_begin[i], unit_begin[i + 1]).
struct ReshapeNewCoreSplit {
    tt::tt_metal::CoreRangeSet all_cores;
    std::vector<tt::tt_metal::CoreCoord> cores;
    std::vector<uint32_t> unit_begin;
};

ReshapeNewCoreSplit split_reshape_units(
    tt::tt_metal::distributed::MeshDevice* device,
    const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grid,
    uint32_t num_units);

// Output page i is input page i (same page sequence, different distribution). Each core moves a contiguous range of
// page chunks through a DFB, a batch per barrier.
struct ReshapeNewPageCopyProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const ReshapeNewParams& operation_attributes, const ReshapeNewInputs& tensor_args, Tensor& output);
};

// Each output unit (a tile, or a chunk of a row-major page) is assembled from element runs of the input. The reader
// walks the reshape geometry and stages the runs with their placements; the writer places them into the unit, fills
// padding, and writes the unit out.
struct ReshapeNewGatherProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const ReshapeNewParams& operation_attributes, const ReshapeNewInputs& tensor_args, Tensor& output);
};

struct ReshapeNewDeviceOperation {
    using operation_attributes_t = ReshapeNewParams;
    using tensor_args_t = ReshapeNewInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;
    using topology_return_value_t = std::vector<tt::tt_metal::TensorTopology>;
    using program_factory_t = std::variant<ReshapeNewPageCopyProgramFactory, ReshapeNewGatherProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    // A reshape relabels Shard(d) to the output dim that starts at the same element offset (#64322).
    static topology_return_value_t compute_output_topologies(const operation_attributes_t&, const tensor_args_t&);
    static tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t> create_op_performance_model(
        const operation_attributes_t&, const tensor_args_t&, const Tensor&);
};

// `output_spec` must hold the same elements as `input` (equal logical volume). `kind` is PageCopy or Gather.
Tensor reshape_new(
    const Tensor& input,
    const tt::tt_metal::TensorSpec& output_spec,
    ttnn::operations::data_movement::reshape_new::ReshapeKind kind,
    uint32_t pad_bits,
    bool fill_padding,
    const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grid);

}  // namespace ttnn::prim

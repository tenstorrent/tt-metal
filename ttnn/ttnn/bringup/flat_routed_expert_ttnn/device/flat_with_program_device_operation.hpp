// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// The flat routed expert with another op's program appended on every chip (one program per chip, so the two run
// together on disjoint cores): the other op comes as a MeshProgramDescriptor (e.g. a Python-built combine), its
// buffers inside the flat expert's arena on its own cells (the caller owns the arena: persistent, so the other op may
// keep state there across calls). Every flat y writer reports each local expert, once its rows have landed, by an
// increment of counts[e] on the collector core (se_cmb_done.hpp, slot order, no go handshake: counts are cumulative,
// the other op compares them with calls x writers).
#pragma once

#include <tuple>
#include <vector>

#include <tt-metalium/experimental/mesh_program_descriptor.hpp>

#include "flat_routed_expert_program_factory.hpp"
#include "flat_routed_expert_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::bringup::flat_routed_expert {

struct FlatWithProgramParams {
    FlatRoutedExpertConfig flat;  // with cmb_rt set
    tt::tt_metal::experimental::MeshProgramDescriptor extra;
    tt::tt_metal::CoreCoord collector{0, 0};  // logical
    uint32_t counts_addr = 0;
    uint32_t writers = 0;
    // every buffer address the program bakes in (the extra program's are its own args): the cache key, plus the
    // caller's key for the extra program, so a cache hit never needs a patch
    std::vector<uint32_t> addrs;
    std::vector<uint32_t> flat_key;
    uint64_t extra_key = 0;

    static constexpr auto attribute_names =
        std::forward_as_tuple("flat_key", "addrs", "extra_key", "writers", "counts_addr", "collector_x", "collector_y");
    auto attribute_values() const {
        return std::forward_as_tuple(
            flat_key,
            addrs,
            extra_key,
            writers,
            counts_addr,
            static_cast<uint32_t>(collector.x),
            static_cast<uint32_t>(collector.y));
    }
};

struct FlatWithProgramSharedVariables {};

struct FlatWithProgramFactory {
    using shared_variables_t = FlatWithProgramSharedVariables;
    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const FlatWithProgramParams& operation_attributes,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const FlatRoutedExpertInputs& tensor_args,
        Tensor& tensor_return_value);

    static void override_runtime_arguments(
        cached_mesh_workload_t&, const FlatWithProgramParams&, const FlatRoutedExpertInputs&, Tensor&) {}
};

struct FlatWithProgramDeviceOperation {
    using operation_attributes_t = FlatWithProgramParams;
    using tensor_args_t = FlatRoutedExpertInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = ttnn::Tensor;
    using program_factory_t = std::variant<FlatWithProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&) {}
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t& t) {
        return t.output.tensor_spec();
    }
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t& t) {
        return t.output;
    }
};

}  // namespace ttnn::operations::bringup::flat_routed_expert

namespace ttnn::prim::bringup {
ttnn::Tensor flat_with_program(
    const ttnn::operations::bringup::flat_routed_expert::FlatWithProgramParams& params,
    const ttnn::operations::bringup::flat_routed_expert::FlatRoutedExpertInputs& inputs);
}  // namespace ttnn::prim::bringup

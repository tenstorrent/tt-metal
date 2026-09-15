// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <tuple>
#include <unordered_map>
#include <variant>

#include <tt-metalium/program.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/mesh_device_operation_adapter.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::transformer::ltx_rope_materialize {

struct LtxRopeMaterializeDeviceOperation {
    struct operation_attributes_t {
        uint32_t sp_axis;
        uint32_t tp_axis;
    };

    struct tensor_args_t {
        const Tensor& compact_self_cos;
        const Tensor& compact_self_sin;
        const Tensor& compact_cross_cos;
        const Tensor& compact_cross_sin;
        const Tensor& metadata;
        const Tensor& self_cos_output;
        const Tensor& self_sin_output;
        const Tensor& cross_cos_output;
        const Tensor& cross_sin_output;
    };

    using spec_return_value_t =
        std::tuple<tt::tt_metal::TensorSpec, tt::tt_metal::TensorSpec, tt::tt_metal::TensorSpec, tt::tt_metal::TensorSpec>;
    using tensor_return_value_t = std::tuple<Tensor, Tensor, Tensor, Tensor>;

    struct SharedVariables {};

    struct MeshWorkloadFactory {
        using shared_variables_t = SharedVariables;
        using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

        static cached_mesh_workload_t create_mesh_workload(
            const operation_attributes_t&,
            const ttnn::MeshCoordinateRangeSet&,
            const tensor_args_t&,
            tensor_return_value_t&);

        static void override_runtime_arguments(
            cached_mesh_workload_t&, const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&);

    private:
        using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

        static cached_program_t create_at(
            const operation_attributes_t&, const ttnn::MeshCoordinate&, const tensor_args_t&, tensor_return_value_t&);
    };

    using program_factory_t = std::variant<MeshWorkloadFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::experimental::transformer::ltx_rope_materialize

namespace ttnn::prim {

std::tuple<Tensor, Tensor, Tensor, Tensor> ltx_rope_materialize(
    const Tensor& compact_self_cos,
    const Tensor& compact_self_sin,
    const Tensor& compact_cross_cos,
    const Tensor& compact_cross_sin,
    const Tensor& metadata,
    const Tensor& self_cos_output,
    const Tensor& self_sin_output,
    const Tensor& cross_cos_output,
    const Tensor& cross_sin_output,
    uint32_t sp_axis,
    uint32_t tp_axis);

}  // namespace ttnn::prim

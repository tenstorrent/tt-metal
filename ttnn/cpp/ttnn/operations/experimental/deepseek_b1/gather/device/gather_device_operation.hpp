// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>
#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::deepseek_b1::gather {

struct GatherDeviceOperation {
    struct operation_attributes_t {
        std::optional<tt::tt_metal::NOC> noc;
    };

    struct tensor_args_t {
        const Tensor& input_tensor;
        std::optional<Tensor> optional_output_tensor;
    };

    using spec_return_value_t = ttnn::TensorSpec;
    using tensor_return_value_t = ttnn::Tensor;

    struct GatherProgramFactory {
        struct shared_variables_t {
            tt::tt_metal::KernelHandle sender_noc0_kernel_id;
            tt::tt_metal::KernelHandle sender_noc1_kernel_id;
            std::vector<tt::tt_metal::CoreCoord> noc_0_cores;
            std::vector<tt::tt_metal::CoreCoord> noc_1_cores;
            uint32_t send_size;
        };

        using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;
        using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

        // Mesh workload pattern
        static cached_mesh_workload_t create_mesh_workload(
            const operation_attributes_t& operation_attributes,
            const ttnn::MeshCoordinateRangeSet& tensor_coords,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);

        static cached_program_t create_at(
            const operation_attributes_t& operation_attributes,
            const ttnn::MeshCoordinate& mesh_coordinate,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);

        static void override_runtime_arguments(
            cached_mesh_workload_t& cached_program,
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    };

    using program_factory_t = std::variant<GatherProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& input_tensor, const Tensor& output_tensor, std::optional<uint32_t> noc);
};

}  // namespace ttnn::operations::experimental::deepseek_b1::gather

// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
///
#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"

#include <tt-metalium/global_semaphore.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>

namespace ttnn {
namespace operations::ccl {

struct ReduceToAllOp {
    struct operation_attributes_t {
        const float scale_fp32;
        const tt::tt_fabric::Topology topology;
        const std::optional<std::vector<ttnn::CoreCoord>> input_mux_cores;
        const std::vector<ttnn::TensorSpec> _input_tensor_spec;

        static constexpr auto attribute_names = std::forward_as_tuple("scale_fp32", "topology", "input_mux_cores");
        auto attribute_values() const { return std::forward_as_tuple(scale_fp32, topology, input_mux_cores); };
    };

    struct tensor_args_t {
        const Tensor input_tensor_l;
        const Tensor input_tensor_ms;  // Combined: col 0 = max, col 1 = sum
        const std::optional<Tensor> optional_output_tensor_l;
        const std::optional<Tensor> optional_fw_intermediate_tensor;
        const std::optional<Tensor> optional_bw_intermediate_tensor;
        const std::optional<Tensor> optional_coord_intermediate_tensor;
        const std::optional<Tensor> optional_aggregator_scratch_tensor;
    };

    using spec_return_value_t = std::array<std::vector<ttnn::TensorSpec>, 2>;
    using tensor_return_value_t = std::array<std::vector<ttnn::Tensor>, 2>;

    struct ReduceToAll {
        struct shared_variables_t {
            tt::tt_metal::KernelHandle reader_kernel;
            std::vector<CoreCoord> worker_cores;

            tt::tt_metal::KernelHandle writer_kernel;

            std::vector<tt::tt_metal::GlobalSemaphore> semaphores;

            // CB handles for aliased buffers (needed for UpdateDynamicCircularBufferAddressAndTotalSize in trace)
            std::optional<tt::tt_metal::CBHandle> cb_local_l_handle;
            std::optional<tt::tt_metal::CBHandle> cb_local_ms_handle;
            std::optional<tt::tt_metal::CBHandle> cb_r1_neighbor_l_handle;
            std::optional<tt::tt_metal::CBHandle> cb_r2_neighbor_l_handle;
            std::optional<tt::tt_metal::CBHandle> cb_l_out_handle;

            // Tile sizes for updating CB total sizes
            uint32_t l_tile_size = 0;
            uint32_t ms_tile_size = 0;
            uint32_t out_tiles = 0;
        };

        using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

        static cached_mesh_workload_t create_mesh_workload(
            const operation_attributes_t& operation_attributes,
            const ttnn::MeshCoordinateRangeSet& tensor_coords,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);

        static ttnn::device_operation::CachedProgram<shared_variables_t> create_at(
            const operation_attributes_t& operation_attributes,
            const ttnn::MeshCoordinate& mesh_coordinate,
            std::optional<MeshCoordinate>& forward_coord,
            std::optional<MeshCoordinate>& backward_coord,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value,
            std::vector<tt::tt_metal::GlobalSemaphore>& semaphores);

        static void override_runtime_arguments(
            cached_mesh_workload_t& cached_workload,
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    };

    using program_factory_t = std::variant<ReduceToAll>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&) {
        return ReduceToAll{};
    };

    static void validate_on_program_cache_miss(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
        validate(operation_attributes, tensor_args);
    };

    static void validate_on_program_cache_hit(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
        validate(operation_attributes, tensor_args);
    };

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

private:
    static void validate(const operation_attributes_t&, const tensor_args_t&);
};

device_operation::CachedProgram<ReduceToAllOp::ReduceToAll::shared_variables_t> reduce_to_all_program_factory(
    const ReduceToAllOp::tensor_args_t& tensor_args,
    const ReduceToAllOp::operation_attributes_t& operation_attributes,
    float scale_fp32,
    const MeshCoordinate& device_coordinate,
    std::optional<MeshCoordinate>& forward_coord,
    std::optional<MeshCoordinate>& backward_coord,
    ReduceToAllOp::tensor_return_value_t& output_tensor,
    std::vector<tt::tt_metal::GlobalSemaphore>& semaphores);

}  // namespace operations::ccl

namespace prim {
ttnn::operations::ccl::ReduceToAllOp::tensor_return_value_t reduce_to_all(
    const Tensor& input_tensor_l,
    const Tensor& input_tensor_ms,  // Combined: col 0 = max, col 1 = sum
    const tt::tt_fabric::Topology& topology,
    float scale_fp32,
    const std::optional<Tensor>& optional_output_tensor_l = std::nullopt,
    const std::optional<Tensor>& optional_fw_intermediate_tensor = std::nullopt,
    const std::optional<Tensor>& optional_bw_intermediate_tensor = std::nullopt,
    const std::optional<Tensor>& optional_coord_intermediate_tensor = std::nullopt,
    const std::optional<std::vector<ttnn::CoreCoord>>& input_mux_cores = std::nullopt,
    const std::optional<Tensor>& optional_aggregator_scratch_tensor = std::nullopt);
}  // namespace prim
}  // namespace ttnn

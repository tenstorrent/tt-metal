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

struct ReduceToRootOp {
    struct operation_attributes_t {
        const MeshCoordinate& root_coord;
        const float scale_fp32;
        const tt::tt_fabric::Topology topology;
        const std::optional<std::vector<ttnn::CoreCoord>> input_mux_cores;

        const std::vector<ttnn::TensorSpec> _input_tensor_spec;

        static constexpr auto attribute_names = std::forward_as_tuple("root_coord", "scale_fp32", "topology");
        auto attribute_values() const { return std::forward_as_tuple(root_coord, scale_fp32, topology); };
    };

    struct tensor_args_t {
        const Tensor input_tensor_l;
        const Tensor input_tensor_s;
        const Tensor input_tensor_m;
        const std::optional<Tensor> optional_output_tensor_l;
        const std::optional<Tensor> optional_output_tensor_s;
        const std::optional<Tensor> optional_output_tensor_m;
        const std::optional<Tensor> optional_intermediate_tensor;
    };

    using spec_return_value_t = std::array<std::vector<ttnn::TensorSpec>, 2>;
    using tensor_return_value_t = std::array<std::vector<ttnn::Tensor>, 2>;

    struct ReduceToRoot {
        struct shared_variables_t {
            tt::tt_metal::KernelHandle send_unary_reader_kernel_id;
            tt::tt_metal::KernelHandle send_unary_writer_kernel_id;
            std::vector<CoreCoord> cores;

            tt::tt_metal::KernelHandle root1_reader_kernel_id;
            tt::tt_metal::KernelHandle root1_writer_kernel_id;

            tt::tt_metal::KernelHandle root2_reader_kernel_id;
            tt::tt_metal::KernelHandle root2_writer_kernel_id;

            tt::tt_metal::KernelHandle compute_kernel_id;

            std::vector<tt::tt_metal::GlobalSemaphore> semaphores;
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
            cached_mesh_workload_t& cached_program,
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    };

    using program_factory_t = std::variant<ReduceToRoot>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&) {
        return ReduceToRoot{};
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

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& input_tensor_l,
        const Tensor& input_tensor_s,
        const Tensor& input_tensor_m,
        const tt::tt_fabric::Topology& topology,
        const MeshCoordinate& root_coord,
        float scale_fp32,
        const std::optional<Tensor>& optional_output_tensor_l = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor_s = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor_m = std::nullopt,
        const std::optional<Tensor>& optional_intermediate_tensor = std::nullopt,
        const std::optional<std::vector<ttnn::CoreCoord>>& input_mux_cores = std::nullopt) {
        return std::make_tuple(
            operation_attributes_t{
                root_coord,
                scale_fp32,
                topology,
                input_mux_cores,
                {input_tensor_l.tensor_spec(), input_tensor_s.tensor_spec(), input_tensor_m.tensor_spec()}},
            tensor_args_t{
                input_tensor_l,
                input_tensor_s,
                input_tensor_m,
                optional_output_tensor_l,
                optional_output_tensor_s,
                optional_output_tensor_m,
                optional_intermediate_tensor});
    };

private:
    static void validate(const operation_attributes_t&, const tensor_args_t&);
};

device_operation::CachedProgram<ReduceToRootOp::ReduceToRoot::shared_variables_t> reduce_to_root_program_factory(
    const ReduceToRootOp::tensor_args_t& tensor_args,
    const ReduceToRootOp::operation_attributes_t& operation_attributes,
    const MeshCoordinate& root_coord,
    float scale_fp32,
    const MeshCoordinate& device_coordinate,
    std::optional<MeshCoordinate>& forward_coord,
    std::optional<MeshCoordinate>& backward_coord,
    ReduceToRootOp::tensor_return_value_t& output_tensor,
    std::vector<tt::tt_metal::GlobalSemaphore>& semaphores);
}  // namespace operations::ccl

namespace prim {
constexpr auto reduce_to_root =
    ttnn::register_operation<"ttnn::prim::reduce_to_root", ttnn::operations::ccl::ReduceToRootOp>();
}  // namespace prim
}  // namespace ttnn

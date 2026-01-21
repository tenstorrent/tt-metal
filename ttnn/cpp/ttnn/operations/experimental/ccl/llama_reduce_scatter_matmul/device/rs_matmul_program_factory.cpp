// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "rs_matmul_op.hpp"
#include <tt-metalium/work_split.hpp>
#include <vector>
#include "ttnn/distributed/types.hpp"
#include "ttnn/operations/experimental/ccl/llama_common.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/shared_with_host/sharded_tensor_addr_gen.hpp"
#include "ttnn/operations/ccl/sharding_addrgen_helper.hpp"
#include <tt-metalium/core_coord.hpp>
#include "ttnn/operations/ccl/common/host/ccl_worker_builder.hpp"
#include <tt-metalium/experimental/fabric/fabric.hpp>

namespace ttnn::operations::experimental::ccl {
Matmul_RS::Matmul_RS_PF::cached_mesh_workload_t Matmul_RS::Matmul_RS_PF::create_mesh_workload(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const tensor_args_t& tensor_args,
    std::vector<Tensor>& tensor_return_value) {
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;
    for (const auto& coord : tensor_coords.coords()) {
        auto cached_program = create_at(operation_attributes, coord, tensor_args, tensor_return_value);
        workload.add_program(ttnn::MeshCoordinateRange(coord), std::move(cached_program.program));
        shared_variables.emplace(coord, std::move(cached_program.shared_variables));
    }
    return cached_mesh_workload_t(std::move(workload), std::move(shared_variables));
}

ttnn::device_operation::CachedProgram<Matmul_RS::Matmul_RS_PF::shared_variables_t> Matmul_RS::Matmul_RS_PF::create_at(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinate& mesh_coordinate,
    const tensor_args_t& tensor_args,
    std::vector<Tensor>& tensor_return_value) {
    tt::tt_metal::Program program{};

    tt::tt_metal::SubDeviceId sub_device_id = operation_attributes.rs_op.subdevice_id.value();
    auto [part_cores, rs_cores] =
        LlamaReduceScatterDeviceOperation::get_rs_core_grids(operation_attributes.rs_op, tensor_args.rs);
    std::optional<CoreRangeSet> reduce_scatter_core_range = rs_cores;
    if (tensor_args.second_weight_tensor.has_value()) {
        ttnn::experimental::ccl::MatmulFusedOpSignaler base_signaler = ttnn::experimental::ccl::MatmulFusedOpSignaler(
            ttnn::experimental::ccl::MatmulFusedOpSignalerType::LLAMA_REDUCE_SCATTER);
        base_signaler.init_llama_rs_cores_rs(rs_cores, program);
        std::optional<ttnn::experimental::ccl::MatmulFusedOpSignaler> fused_op_signaler = base_signaler;
        auto reduce_scatter_sv = LlamaReduceScatterDeviceOperation::LlamaReduceScatterAdd::create_at_program_processing(
            operation_attributes.rs_op,
            mesh_coordinate,
            tensor_args.rs,
            tensor_return_value.at(2),
            program,
            fused_op_signaler);
        auto matmul_sv = ttnn::prim::matmul_multi_core_reuse_mcast_1d_optimized_helper(
            program,
            tensor_args.matmul.input_tensor,
            {tensor_args.matmul.weight_tensor, tensor_args.second_weight_tensor.value()},
            std::nullopt /*bias*/,
            {tensor_return_value.at(0), tensor_return_value.at(1)},
            operation_attributes.matmul.bcast_batch.value(),
            operation_attributes.matmul.compute_kernel_config.value(),
            operation_attributes.matmul.program_config.value(),
            operation_attributes.matmul.untilize_out,
            fused_op_signaler,
            operation_attributes.matmul.global_cb,
            sub_device_id /*sub_device_id*/,
            tt::CBIndex::c_6 /*start cb index*/,
            reduce_scatter_core_range);
        return {std::move(program), shared_variables_t{reduce_scatter_sv, matmul_sv}};
    }
    std::optional<ttnn::experimental::ccl::MatmulFusedOpSignaler> fused_op_signaler = std::nullopt;
    auto reduce_scatter_sv = LlamaReduceScatterDeviceOperation::LlamaReduceScatterAdd::create_at_program_processing(
        operation_attributes.rs_op,
        mesh_coordinate,
        tensor_args.rs,
        tensor_return_value.at(1),
        program,
        fused_op_signaler);
    auto matmul_sv = ttnn::prim::matmul_multi_core_reuse_mcast_1d_optimized_helper(
        program,
        tensor_args.matmul.input_tensor,
        {tensor_args.matmul.weight_tensor},
        std::nullopt /*bias*/,
        {tensor_return_value.at(0)},
        operation_attributes.matmul.bcast_batch.value(),
        operation_attributes.matmul.compute_kernel_config.value(),
        operation_attributes.matmul.program_config.value(),
        operation_attributes.matmul.untilize_out,
        fused_op_signaler,
        operation_attributes.matmul.global_cb,
        sub_device_id /*sub_device_id*/,
        tt::CBIndex::c_6 /*start cb index*/,
        reduce_scatter_core_range);
    return {std::move(program), shared_variables_t{reduce_scatter_sv, matmul_sv}};
}

void Matmul_RS::Matmul_RS_PF::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    std::vector<Tensor>& tensor_return_value) {
    if (tensor_args.second_weight_tensor.has_value()) {
        for (auto& [range, program] : cached_workload.workload.get_programs()) {
            const auto& shared_variables = cached_workload.shared_variables.at(range);
            LlamaReduceScatterDeviceOperation::LlamaReduceScatterAdd::override_runtime_arguments_per_program(
                shared_variables.rs_shared_vars,
                program,
                operation_attributes.rs_op,
                tensor_args.rs,
                tensor_return_value.at(2));
            ttnn::prim::reuse_mcast_1d_optimized_helpers::override_program_parameters(
                shared_variables.matmul_shared_vars,
                operation_attributes.matmul.global_cb,
                program,
                {{tensor_args.matmul.input_tensor,
                  tensor_args.matmul.weight_tensor,
                  tensor_args.second_weight_tensor.value()},
                 {}},
                {tensor_return_value.at(0), tensor_return_value.at(1)});
        }
    } else {
        for (auto& [range, program] : cached_workload.workload.get_programs()) {
            const auto& shared_variables = cached_workload.shared_variables.at(range);
            LlamaReduceScatterDeviceOperation::LlamaReduceScatterAdd::override_runtime_arguments_per_program(
                shared_variables.rs_shared_vars,
                program,
                operation_attributes.rs_op,
                tensor_args.rs,
                tensor_return_value.at(1));
            ttnn::prim::reuse_mcast_1d_optimized_helpers::override_program_parameters(
                shared_variables.matmul_shared_vars,
                operation_attributes.matmul.global_cb,
                program,
                {{tensor_args.matmul.input_tensor, tensor_args.matmul.weight_tensor}, {}},
                {tensor_return_value.at(0)});
        }
    }
}
}  // namespace ttnn::operations::experimental::ccl

// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "rs_matmul_op.hpp"
#include <tt-metalium/work_split.hpp>
#include <vector>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/device_pool.hpp>
#include "ttnn/distributed/types.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/experimental/ccl/all_gather_async/device/all_gather_async_op.hpp"
#include "cpp/ttnn/operations/ccl/shared_with_host/sharded_tensor_addr_gen.hpp"
#include "cpp/ttnn/operations/ccl/sharding_addrgen_helper.hpp"
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/erisc_datamover_builder.hpp>
#include "cpp/ttnn/operations/ccl/common/host/ccl_worker_builder.hpp"
#include <tt-metalium/fabric.hpp>
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
    std::optional<ttnn::experimental::ccl::MatmulFusedOpSignaler> empty_fused_op_signaler;
    tt::tt_metal::SubDeviceId sub_device_id = operation_attributes.rs_op.subdevice_id.value();
    CoreRangeSet rs_cores =
        CoreRangeSet(std::set{::CoreRange{{1, 1}, {3, 2}}, ::CoreRange{{1, 3}, {2, 3}}, ::CoreRange{{1, 6}, {2, 7}}});
    std::optional<CoreRangeSet> optional_core_range = rs_cores;
    return {
        std::move(program),
        shared_variables_t{
            LlamaReduceScatterDeviceOperation::LlamaReduceScatterAdd::create_at_program_processing(
                operation_attributes.rs_op, mesh_coordinate, tensor_args.rs, tensor_return_value.at(1), program),
            matmul::matmul_multi_core_reuse_mcast_1d_optimized_helper(
                program,
                tensor_args.matmul.input_tensor,
                {tensor_args.matmul.weight_tensor},
                std::nullopt /*bias*/,
                {tensor_return_value.at(0)},
                operation_attributes.matmul.bcast_batch.value(),
                operation_attributes.matmul.compute_kernel_config.value(),
                operation_attributes.matmul.program_config.value(),
                operation_attributes.matmul.untilize_out,
                empty_fused_op_signaler,
                operation_attributes.matmul.global_cb,
                sub_device_id /*sub_device_id*/,
                tt::CBIndex::c_6 /*start cb index*/,
                optional_core_range)}};
}

void Matmul_RS::Matmul_RS_PF::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    std::vector<Tensor>& tensor_return_value) {
    for (auto& [range, program] : cached_workload.workload.get_programs()) {
        const auto& shared_variables = cached_workload.shared_variables.at(range);
        LlamaReduceScatterDeviceOperation::LlamaReduceScatterAdd::override_runtime_arguments_per_program(
            shared_variables.rs_shared_vars,
            program,
            operation_attributes.rs_op,
            tensor_args.rs,
            tensor_return_value.at(1));
        reuse_mcast_1d_optimized_helpers::override_program_parameters(
            shared_variables.matmul_shared_vars,
            &operation_attributes.matmul,
            program,
            {tensor_args.matmul.input_tensor, tensor_args.matmul.weight_tensor},
            {},
            {tensor_return_value.at(0)});
    }
}
}  // namespace ttnn::operations::experimental::ccl

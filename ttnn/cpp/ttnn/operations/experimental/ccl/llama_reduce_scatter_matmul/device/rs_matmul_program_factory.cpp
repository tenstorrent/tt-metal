// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
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
#include "ttnn/operations/ccl/common/host/ccl_worker_builder.hpp"
#include "ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_1d_program_factory.hpp"
#include <tt-metalium/experimental/fabric/fabric.hpp>

namespace ttnn::operations::experimental::ccl {

// Contract-2 (WorkloadDescriptor) factory.  Composes a single ProgramDescriptor
// per mesh coord that runs the reduce-scatter and matmul halves on
// non-overlapping core ranges (rs on rs_cores, matmul restricted to the
// complement via `reduce_scatter_core_range`).  Both halves share a
// MatmulFusedOpSignaler so the matmul master can signal the rs reader cores —
// the signaler's rs_semaphore is allocated first (so its ID is stable for the
// downstream push_llama_rs_rt_args_for_rs() calls), then the rs builder appends
// its own kernels/CBs/local_semaphore (which uses find_available_semaphore_id
// to avoid colliding with the rs_semaphore on overlapping cores), and finally
// the matmul descriptor helper appends its kernels/CBs and the matmul
// privileged semaphore.
tt::tt_metal::WorkloadDescriptor Matmul_RS::Matmul_RS_PF::create_workload_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    std::vector<Tensor>& tensor_return_value,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    tt::tt_metal::WorkloadDescriptor workload_descriptor;

    tt::tt_metal::SubDeviceId sub_device_id = operation_attributes.rs_op.subdevice_id.value();
    auto [part_cores, rs_cores] =
        LlamaReduceScatterDeviceOperation::get_rs_core_grids(operation_attributes.rs_op, tensor_args.rs);
    std::optional<CoreRangeSet> reduce_scatter_core_range = rs_cores;

    for (const auto& coord : tensor_coords.coords()) {
        tt::tt_metal::ProgramDescriptor desc;

        if (tensor_args.second_weight_tensor.has_value()) {
            // Two-weight path: matmul produces two outputs, and the reduce
            // scatter consumes the third tensor_return_value entry.
            ttnn::experimental::ccl::MatmulFusedOpSignaler base_signaler(
                ttnn::experimental::ccl::MatmulFusedOpSignalerType::LLAMA_REDUCE_SCATTER);
            base_signaler.init_llama_rs_cores_rs(rs_cores, desc);
            std::optional<ttnn::experimental::ccl::MatmulFusedOpSignaler> fused_op_signaler = base_signaler;

            LlamaReduceScatterDeviceOperation::LlamaReduceScatterAdd::create_at_program_processing_descriptor(
                desc, operation_attributes.rs_op, coord, tensor_args.rs, tensor_return_value.at(2), fused_op_signaler);

            ttnn::prim::matmul_multi_core_reuse_mcast_1d_optimized_helper_descriptor(
                desc,
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
        } else {
            // Single-weight path: matmul produces one output, and the reduce
            // scatter consumes the second tensor_return_value entry.  No
            // signaler is plumbed here because the legacy single-weight path
            // also runs without one (matches the legacy create_at body).
            std::optional<ttnn::experimental::ccl::MatmulFusedOpSignaler> fused_op_signaler = std::nullopt;

            LlamaReduceScatterDeviceOperation::LlamaReduceScatterAdd::create_at_program_processing_descriptor(
                desc, operation_attributes.rs_op, coord, tensor_args.rs, tensor_return_value.at(1), fused_op_signaler);

            ttnn::prim::matmul_multi_core_reuse_mcast_1d_optimized_helper_descriptor(
                desc,
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
        }

        workload_descriptor.programs.push_back({ttnn::MeshCoordinateRange(coord), std::move(desc)});
    }

    return workload_descriptor;
}
}  // namespace ttnn::operations::experimental::ccl

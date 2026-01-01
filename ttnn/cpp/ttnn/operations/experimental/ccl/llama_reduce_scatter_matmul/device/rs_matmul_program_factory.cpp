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
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/operations/experimental/matmul/ccl_fusion/device/ccl_fusion_gather_in0.hpp"

namespace ttnn::operations::experimental::ccl {

namespace {

constexpr LlamaReduceScatterDeviceOperation::LlamaReduceScatterAdd::descriptor_artifacts_t kReduceScatterKernels{
    .reader_kernel_index = LlamaReduceScatterDeviceOperation::LlamaReduceScatterAdd::kReaderKernelIdx,
    .writer_kernel_index = LlamaReduceScatterDeviceOperation::LlamaReduceScatterAdd::kWriterKernelIdx};

tt::tt_metal::ProgramDescriptor build_program_descriptor(
    const Matmul_RS::operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinate& mesh_coordinate,
    const Matmul_RS::tensor_args_t& tensor_args,
    std::vector<Tensor>& tensor_return_value) {
    tt::tt_metal::ProgramDescriptor desc;

    tt::tt_metal::SubDeviceId sub_device_id = operation_attributes.rs_op.subdevice_id.value();
    auto [part_cores, rs_cores] =
        LlamaReduceScatterDeviceOperation::get_rs_core_grids(operation_attributes.rs_op, tensor_args.rs);
    // The fused matmul is built through the gather_in0 helper, which has no PrefetcherPipe transport: it would build a
    // plain in1 buffer and read the weight from DRAM.
    TT_FATAL(
        operation_attributes.matmul.prefetcher_pipes.empty(),
        "The fused reduce-scatter + matmul path does not support prefetcher_pipes in1 delivery");
    std::optional<CoreRangeSet> reduce_scatter_core_range = rs_cores;
    const bool two_weights = tensor_args.second_weight_tensor.has_value();

    std::optional<ttnn::experimental::ccl::MatmulFusedOpSignaler> fused_op_signaler = std::nullopt;
    if (two_weights) {
        ttnn::experimental::ccl::MatmulFusedOpSignaler base_signaler = ttnn::experimental::ccl::MatmulFusedOpSignaler(
            ttnn::experimental::ccl::MatmulFusedOpSignalerType::LLAMA_REDUCE_SCATTER);
        base_signaler.init_llama_rs_cores_rs(rs_cores, desc);
        fused_op_signaler = base_signaler;
    }
    const auto reduce_scatter_kernels =
        LlamaReduceScatterDeviceOperation::LlamaReduceScatterAdd::create_at_program_descriptor_processing(
            operation_attributes.rs_op,
            mesh_coordinate,
            tensor_args.rs,
            tensor_return_value.at(two_weights ? 2 : 1),
            desc,
            fused_op_signaler);
    TT_FATAL(
        reduce_scatter_kernels.reader_kernel_index == kReduceScatterKernels.reader_kernel_index &&
            reduce_scatter_kernels.writer_kernel_index == kReduceScatterKernels.writer_kernel_index,
        "llama_rs_matmul: the reduce-scatter reader/writer must be the first two kernels (got {}, {}); the cache-hit "
        "override depends on it",
        reduce_scatter_kernels.reader_kernel_index,
        reduce_scatter_kernels.writer_kernel_index);

    std::vector<Tensor> weights = {tensor_args.matmul.weight_tensor};
    std::vector<Tensor> outputs = {tensor_return_value.at(0)};
    if (two_weights) {
        weights.push_back(tensor_args.second_weight_tensor.value());
        outputs.push_back(tensor_return_value.at(1));
    }
    ttnn::prim::ccl_fusion::matmul_multi_core_reuse_mcast_1d_gather_in0_helper(
        desc,
        tensor_args.matmul.input_tensor,
        weights,
        std::nullopt /*bias*/,
        outputs,
        operation_attributes.matmul.bcast_batch.value(),
        operation_attributes.matmul.compute_kernel_config.value(),
        operation_attributes.matmul.program_config.value(),
        operation_attributes.matmul.untilize_out,
        fused_op_signaler,
        operation_attributes.matmul.global_cb.has_value() ? &*operation_attributes.matmul.global_cb : nullptr,
        sub_device_id /*sub_device_id*/,
        tt::CBIndex::c_6 /*start cb index*/,
        reduce_scatter_core_range);
    return desc;
}

}  // namespace

tt::tt_metal::WorkloadDescriptor Matmul_RS::Matmul_RS_PF::create_workload_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    std::vector<Tensor>& tensor_return_value,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    tt::tt_metal::WorkloadDescriptor workload;
    const auto coords = tensor_coords.coords();
    workload.programs.reserve(coords.size());
    for (const auto& coord : coords) {
        workload.programs.push_back(
            {ttnn::MeshCoordinateRange(coord),
             build_program_descriptor(operation_attributes, coord, tensor_args, tensor_return_value)});
    }
    return workload;
}

Matmul_RS::Matmul_RS_MeshWorkloadFactory::cached_mesh_workload_t
Matmul_RS::Matmul_RS_MeshWorkloadFactory::create_mesh_workload(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const tensor_args_t& tensor_args,
    std::vector<Tensor>& tensor_return_value) {
    auto cached = descriptor_adapter_t::create_mesh_workload(
        operation_attributes, tensor_coords, tensor_args, tensor_return_value);
    // resolve_bindings produces NO bindings when a buffer repeats within the inputs (the same tensor passed as both
    // weights, say): it cannot tell the two slots apart, and the WorkloadDescriptor adapter has no slow-path
    // rebuild, so such a call would run its program-cache hits on the first call's addresses. Refuse it here, on
    // the miss, rather than patch nothing later.
    for (const auto& [range, shared] : cached.shared_variables) {
        TT_FATAL(
            !shared.resolved_bindings.empty(),
            "llama_rs_matmul: two of the input tensors share a buffer (the same tensor passed twice?); the fused "
            "program cannot be re-addressed on a program-cache hit, so this call is not supported");
    }
    return cached;
}

void Matmul_RS::Matmul_RS_MeshWorkloadFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    std::vector<Tensor>& tensor_return_value) {
    // Tensor addresses: reduce-scatter input/output/packet-buffer CBs, matmul in0/output CBs and in1 address, through
    // the bindings resolve_bindings produced on the miss (create_mesh_workload refuses a call that gets none).
    descriptor_adapter_t::apply_descriptor(cached_workload, operation_attributes, tensor_args, tensor_return_value);

    // The caller-owned cross-device GlobalSemaphore.
    for (auto& [range, program] : cached_workload.workload.get_programs()) {
        LlamaReduceScatterDeviceOperation::LlamaReduceScatterAdd::apply_cross_device_semaphore(
            program, kReduceScatterKernels, operation_attributes.rs_op, tensor_args.rs);
    }
}

static_assert(ttnn::device_operation::MeshWorkloadFactoryConcept<Matmul_RS::Matmul_RS_MeshWorkloadFactory>);

}  // namespace ttnn::operations::experimental::ccl

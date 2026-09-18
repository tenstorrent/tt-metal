// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/ccl/matmul_reduce_scatter_sp_async/matmul_reduce_scatter_sp_async.hpp"

#include <cstdlib>
#include <cstring>

#include <tt-metalium/math.hpp>

#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/common/host/moe_utils.hpp"
#include "ttnn/operations/experimental/ccl/matmul_reduce_scatter_sp_async/device/matmul_reduce_scatter_sp_async_device_operation.hpp"

namespace ttnn::experimental {

Tensor matmul_reduce_scatter_sp_async(
    const Tensor& input,
    const Tensor& weight,
    const uint32_t cluster_axis,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    const std::optional<GlobalSemaphore>& barrier_semaphore,
    const bool transpose_b,
    const std::optional<uint32_t> num_links,
    const ttnn::ccl::Topology topology,
    const uint32_t ccl_core_rows,
    const std::optional<uint32_t> num_workers_per_link,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<const DataType> dtype,
    const std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<const operations::matmul::MatmulProgramConfig>& program_config,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id,
    const bool debug_serialize_reduce_scatter) {
    auto* mesh_device = input.device();
    TT_FATAL(mesh_device != nullptr, "Mesh device is required for matmul_reduce_scatter_sp_async");

    const uint32_t resolved_num_links =
        num_links.value_or(ttnn::operations::ccl::common::get_num_links(*mesh_device, cluster_axis));
    const uint32_t ring_size = ttnn::ccl::get_topological_dimension(input, cluster_axis);
    TT_FATAL(
        ring_size > 1,
        "matmul_reduce_scatter_sp_async needs > 1 device along cluster_axis {}, got {}",
        cluster_axis,
        ring_size);
    // Ring is demoted to Linear when the axis is not wrap-wired (as the standalone reduce_scatter_minimal_async).
    const ttnn::ccl::Topology usable_topology = ttnn::ccl::get_usable_topology(input, topology, cluster_axis);
    const uint32_t resolved_workers_per_link = num_workers_per_link.value_or(
        prim::sp_default_reduce_scatter_workers(input, usable_topology, resolved_num_links, ccl_core_rows));

    // Matmul numerics: replicate ttnn::matmul's defaults (HiFi2, no approx, fp32 accumulation only for fp32
    // outputs, L1 accumulation otherwise) so that fused == unfused bitwise for the same in0_block_w. Passing a
    // program config to create_matmul_attributes would drop the fidelity to LoFi, so resolve it here explicitly.
    const DataType output_dtype = dtype.value_or(input.dtype());
    const bool is_f32 = output_dtype == DataType::FLOAT32;
    const DeviceComputeKernelConfig matmul_compute_kernel_config = init_device_compute_kernel_config(
        mesh_device->arch(),
        compute_kernel_config,
        tt::tt_metal::MathFidelity::HiFi2,
        /*default_approx_mode=*/false,
        /*default_fp32_acc=*/is_f32,
        /*default_l1_acc=*/!is_f32);
    // Reduce-scatter accumulation: fp32 dest-acc for fp32 data, as the standalone op.
    const auto reduce_scatter_compute_kernel_config =
        ttnn::ccl::resolve_fp32_acc_compute_kernel_config(std::nullopt, output_dtype);

    // Perf-decomposition knob: TT_SP_IN1_STREAM=1 streams the weight per sub-batch instead of keeping it resident.
    const char* in1_stream_env = std::getenv("TT_SP_IN1_STREAM");
    const bool in1_resident =
        !(in1_stream_env != nullptr && std::strcmp(in1_stream_env, "0") != 0 && std::strcmp(in1_stream_env, "") != 0);
    auto outputs = ttnn::prim::matmul_reduce_scatter_sp_async(
        input,
        weight,
        cluster_axis,
        multi_device_global_semaphore,
        barrier_semaphore,
        transpose_b,
        resolved_num_links,
        ring_size,
        usable_topology,
        ccl_core_rows,
        resolved_workers_per_link,
        memory_config,
        output_dtype,
        matmul_compute_kernel_config,
        reduce_scatter_compute_kernel_config,
        program_config,
        sub_device_id,
        debug_serialize_reduce_scatter,
        in1_resident);
    // Dropping the vector releases the mm partial and the RS intermediates; only the RS output is returned.
    return outputs.at(prim::kRsOutputIdx);
}

std::vector<uint32_t> matmul_reduce_scatter_sp_rs_first_touch_order(
    const ttnn::ccl::Topology topology, const uint32_t ring_size, const uint32_t ring_index) {
    return prim::rs_first_touch_order(topology, ring_size, ring_index);
}

}  // namespace ttnn::experimental

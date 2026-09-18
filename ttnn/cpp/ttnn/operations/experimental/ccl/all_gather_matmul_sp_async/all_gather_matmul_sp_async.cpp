// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/ccl/all_gather_matmul_sp_async/all_gather_matmul_sp_async.hpp"

#include <cstdlib>
#include <cstring>

#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/common/host/moe_utils.hpp"
#include "ttnn/operations/experimental/ccl/all_gather_matmul_sp_async/device/all_gather_matmul_sp_async_device_operation.hpp"
#include "ttnn/operations/experimental/ccl/sp_matmul_fusion_common/sp_matmul_fusion_common.hpp"

namespace ttnn::experimental {

std::vector<Tensor> all_gather_matmul_sp_async(
    const Tensor& input,
    const Tensor& weight,
    const uint32_t cluster_axis,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    const std::optional<GlobalSemaphore>& barrier_semaphore,
    const bool transpose_b,
    const std::optional<const Tensor>& bias,
    const std::optional<uint32_t> num_links,
    const ttnn::ccl::Topology topology,
    const uint32_t ccl_core_rows,
    const std::optional<uint32_t> num_workers_per_link,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<const DataType> dtype,
    const std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<const operations::matmul::MatmulProgramConfig>& program_config,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id) {
    auto* mesh_device = input.device();
    TT_FATAL(mesh_device != nullptr, "Mesh device is required for all_gather_matmul_sp_async");

    const uint32_t resolved_num_links =
        num_links.value_or(ttnn::operations::ccl::common::get_num_links(*mesh_device, cluster_axis));
    const uint32_t ring_size = ttnn::ccl::get_topological_dimension(input, cluster_axis);
    TT_FATAL(
        ring_size > 1,
        "all_gather_matmul_sp_async needs > 1 device along cluster_axis {}, got {}",
        cluster_axis,
        ring_size);
    // Ring is demoted to Linear when the axis is not wrap-wired (as the standalone all_gather_async).
    const ttnn::ccl::Topology usable_topology = ttnn::ccl::get_usable_topology(input, topology, cluster_axis);
    const uint32_t resolved_workers_per_link = num_workers_per_link.value_or(
        prim::sp_default_all_gather_workers(input, ring_size, usable_topology, resolved_num_links, ccl_core_rows));
    // Perf-decomposition knob (see AllGatherMatmulSpAsyncParams::debug_serialize_ag); part of the program hash.
    const char* serialize_env = std::getenv("TT_SP_AG_MM_SERIALIZE");
    const bool debug_serialize_ag =
        serialize_env != nullptr && std::strcmp(serialize_env, "0") != 0 && std::strcmp(serialize_env, "") != 0;
    // Perf-decomposition knob: TT_SP_AG_SIGNAL_LATE=1 restores the historical fused-AG signal timing (a forwarded
    // slice is signalled after it has been forwarded, one slice-time after it landed).
    const char* signal_late_env = std::getenv("TT_SP_AG_SIGNAL_LATE");
    const bool ag_signal_on_receive =
        !(signal_late_env != nullptr && std::strcmp(signal_late_env, "0") != 0 && std::strcmp(signal_late_env, "") != 0);
    // Perf-decomposition knob: TT_SP_IN1_STREAM=1 streams the weight per sub-batch instead of keeping it resident.
    const char* in1_stream_env = std::getenv("TT_SP_IN1_STREAM");
    const bool in1_resident =
        !(in1_stream_env != nullptr && std::strcmp(in1_stream_env, "0") != 0 && std::strcmp(in1_stream_env, "") != 0);

    // Matmul numerics: ttnn::matmul's defaults (HiFi2, no approx, fp32 accumulation only for fp32 outputs, L1
    // accumulation otherwise). Passing a program config to create_matmul_attributes would drop the fidelity to
    // LoFi, so resolve it here explicitly (same choice as matmul_reduce_scatter_sp_async).
    const DataType output_dtype = dtype.value_or(input.dtype());
    const bool is_f32 = output_dtype == DataType::FLOAT32;
    const DeviceComputeKernelConfig matmul_compute_kernel_config = init_device_compute_kernel_config(
        mesh_device->arch(),
        compute_kernel_config,
        tt::tt_metal::MathFidelity::HiFi2,
        /*default_approx_mode=*/false,
        /*default_fp32_acc=*/is_f32,
        /*default_l1_acc=*/!is_f32);

    return ttnn::prim::all_gather_matmul_sp_async(
        input,
        weight,
        cluster_axis,
        multi_device_global_semaphore,
        barrier_semaphore,
        transpose_b,
        bias,
        resolved_num_links,
        usable_topology,
        ccl_core_rows,
        resolved_workers_per_link,
        memory_config,
        output_dtype,
        matmul_compute_kernel_config,
        program_config,
        sub_device_id,
        debug_serialize_ag,
        ag_signal_on_receive,
        in1_resident);
}

std::vector<std::vector<uint32_t>> all_gather_matmul_sp_ag_schedule(
    const ttnn::ccl::Topology topology, const uint32_t ring_size, const uint32_t ring_index, const uint32_t batch) {
    std::vector<std::vector<uint32_t>> rows;
    for (const auto& e : ttnn::experimental::ccl::sp_ag_schedule(topology, ring_size, ring_index, batch)) {
        rows.push_back({e.in0_idx, e.out_idx, e.wait_dir, e.wait_count, static_cast<uint32_t>(e.is_local)});
    }
    return rows;
}

}  // namespace ttnn::experimental

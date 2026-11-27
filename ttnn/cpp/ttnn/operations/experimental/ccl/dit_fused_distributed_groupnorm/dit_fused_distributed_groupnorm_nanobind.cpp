// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dit_fused_distributed_groupnorm_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"
#include "ttnn/operations/experimental/ccl/dit_fused_distributed_groupnorm/dit_fused_distributed_groupnorm.hpp"

namespace ttnn::operations::experimental::ccl {

// Accepts the activation as a string ("silu", "gelu", ...) like dit_rms_norm_unary_fused does,
// so the model layer does not have to construct a UnaryWithParam.
ttnn::Tensor dit_fused_distributed_groupnorm_wrapper(
    const ttnn::Tensor& input_tensor,
    int num_groups,
    float epsilon,
    uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    ttnn::ccl::Topology topology,
    const std::optional<ttnn::Tensor>& input_mask,
    const std::optional<ttnn::Tensor>& weight,
    const std::optional<ttnn::Tensor>& bias,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<ttnn::Tensor>& persistent_output_buffer,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id,
    const std::optional<std::string>& activation) {
    std::optional<ttnn::operations::unary::UnaryWithParam> act_param = std::nullopt;
    if (activation.has_value()) {
        act_param = ttnn::operations::unary::utils::string_to_unary_with_param(activation.value());
    }
    return ttnn::experimental::dit_fused_distributed_groupnorm(
        input_tensor,
        num_groups,
        epsilon,
        cluster_axis,
        mesh_device,
        multi_device_global_semaphore,
        topology,
        input_mask,
        weight,
        bias,
        memory_config,
        compute_kernel_config,
        persistent_output_buffer,
        subdevice_id,
        act_param);
}

void bind_dit_fused_distributed_groupnorm(nb::module_& mod) {
    ttnn::bind_function<"dit_fused_distributed_groupnorm", "ttnn.experimental.">(
        mod,
        R"doc(
            Fused distributed GroupNorm for spatially sharded activations.

            Same contract as ``ttnn.group_norm`` (Welford, DRAM-packed RM γ/β +
            ``input_mask``), plus fabric all-gather of per-group stats on
            ``cluster_axis`` (PRE → AG → POST). When mesh width on that axis is
            1, runs local PRE+POST with no fabric. The all-gather always uses a
            single fabric link.

            ``activation`` optionally fuses a unary activation (e.g. ``"silu"``)
            into the output stage, equivalent to
            ``ttnn.<activation>(dit_fused_distributed_groupnorm(x, ...))`` but
            without the extra full-tensor DRAM read/write. The activation
            consumes the fp32 DEST value (after gamma/beta), so it is applied
            before the output is rounded to the output dtype.
        )doc",
        &dit_fused_distributed_groupnorm_wrapper,
        nb::arg("input_tensor"),
        nb::kw_only(),
        nb::arg("num_groups"),
        nb::arg("epsilon") = 1e-5,
        nb::arg("cluster_axis"),
        nb::arg("mesh_device"),
        nb::arg("multi_device_global_semaphore"),
        nb::arg("topology") = ttnn::ccl::Topology::Ring,
        nb::arg("input_mask") = nb::none(),
        nb::arg("weight") = nb::none(),
        nb::arg("bias") = nb::none(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none(),
        nb::arg("persistent_output_buffer") = nb::none(),
        nb::arg("subdevice_id") = nb::none(),
        nb::arg("activation") = nb::none());

    ttnn::bind_function<"dit_fused_distributed_groupnorm_create_stats_buffer", "ttnn.experimental.">(
        mod,
        R"doc(
            Allocate the persistent DRAM stats scratch buffer for
            `dit_fused_distributed_groupnorm`'s all-gather path (cluster width > 1).

            Returns None when cluster width is 1 (no AG). The caller must hold the
            tensor across launches and pass it via `persistent_output_buffer`.
            Always sized for a single fabric link.
        )doc",
        &ttnn::experimental::dit_fused_distributed_groupnorm_create_stats_buffer,
        nb::arg("input_tensor"),
        nb::arg("num_groups"),
        nb::arg("cluster_axis"),
        nb::arg("mesh_device"));
}

}  // namespace ttnn::operations::experimental::ccl

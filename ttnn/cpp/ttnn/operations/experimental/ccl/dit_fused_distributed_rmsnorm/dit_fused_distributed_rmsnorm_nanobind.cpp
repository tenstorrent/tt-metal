// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dit_fused_distributed_rmsnorm_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/ccl/dit_fused_distributed_rmsnorm/dit_fused_distributed_rmsnorm.hpp"

namespace ttnn::operations::experimental::ccl {

void bind_dit_fused_distributed_rmsnorm(nb::module_& mod) {
    ttnn::bind_function<"dit_fused_distributed_rmsnorm", "ttnn.experimental.">(
        mod,
        R"doc(
            Fused distributed RMSNorm for DiT attention.

            One fused device op (per-chip reader/compute/writer with a fabric-forwarder
            all-gather): per-row partial sum-of-squares, all-gather of the partial stats
            across `cluster_axis`, then finalize x * rsqrt(E[x^2] + eps) with optional head
            split, RoPE, and output-dtype cast.
        )doc",
        &ttnn::experimental::dit_fused_distributed_rmsnorm,
        nb::arg("input_tensor"),
        nb::arg("cluster_axis"),
        nb::arg("mesh_device"),
        nb::arg("multi_device_global_semaphore"),
        nb::kw_only(),
        nb::arg("topology") = ttnn::ccl::Topology::Ring,
        nb::arg("epsilon") = 1e-5,
        nb::arg("num_heads_per_device") = 1,
        nb::arg("per_head_norm") = false,
        nb::arg("weight") = nb::none(),
        nb::arg("bias") = nb::none(),
        nb::arg("transformation_mat") = nb::none(),
        nb::arg("rope_cos") = nb::none(),
        nb::arg("rope_sin") = nb::none(),
        nb::arg("dtype") = nb::none(),
        nb::arg("persistent_output_buffer") = nb::none(),
        nb::arg("num_preferred_links") = nb::none(),
        nb::arg("subdevice_id") = nb::none(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none());

    ttnn::bind_function<"dit_fused_distributed_layernorm", "ttnn.experimental.">(
        mod,
        R"doc(
            Fused distributed Welford LayerNorm for Wan2.2 attention.

            Same fused device op / fabric all-gather as `dit_fused_distributed_rmsnorm`,
            but computes a numerically-stable Welford mean/variance and applies
            (x - mean) * rsqrt(var + eps) with optional weight/bias, head split, RoPE, and
            output-dtype cast. Only the fused device op path exists for LayerNorm.

            Pass `reciprocals` (== ttnn.create_layer_norm_reciprocals) to let the Welford
            LLK do an array load of 1/(N+1) instead of a soft-float divide per sample.
        )doc",
        &ttnn::experimental::dit_fused_distributed_layernorm,
        nb::arg("input_tensor"),
        nb::arg("cluster_axis"),
        nb::arg("mesh_device"),
        nb::arg("multi_device_global_semaphore"),
        nb::kw_only(),
        nb::arg("topology") = ttnn::ccl::Topology::Ring,
        nb::arg("epsilon") = 1e-5,
        nb::arg("num_heads_per_device") = 1,
        nb::arg("weight") = nb::none(),
        nb::arg("bias") = nb::none(),
        nb::arg("transformation_mat") = nb::none(),
        nb::arg("rope_cos") = nb::none(),
        nb::arg("rope_sin") = nb::none(),
        nb::arg("dtype") = nb::none(),
        nb::arg("persistent_output_buffer") = nb::none(),
        nb::arg("num_preferred_links") = nb::none(),
        nb::arg("subdevice_id") = nb::none(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none(),
        nb::arg("reciprocals") = nb::none());

    ttnn::bind_function<"dit_fused_distributed_rmsnorm_create_stats_buffer", "ttnn.experimental.">(
        mod,
        R"doc(
            Allocate the persistent stats DRAM scratch buffer required by
            `dit_fused_distributed_rmsnorm`'s all-gather path (TP>1, whole-row norm).
            Returns None when the op reduces locally and needs no scratch (TP=1 or
            per_head_norm). Hold the returned tensor across launches and pass it in via
            the `persistent_output_buffer` kwarg.
        )doc",
        &ttnn::experimental::dit_fused_distributed_rmsnorm_create_stats_buffer,
        nb::arg("input_tensor"),
        nb::arg("cluster_axis"),
        nb::arg("mesh_device"),
        nb::kw_only(),
        nb::arg("num_heads_per_device") = 1,
        nb::arg("per_head_norm") = false,
        nb::arg("num_links") = 1,
        nb::arg("weight") = nb::none(),
        nb::arg("transformation_mat") = nb::none(),
        nb::arg("rope_cos") = nb::none(),
        nb::arg("rope_sin") = nb::none());

    ttnn::bind_function<"dit_fused_distributed_layernorm_create_stats_buffer", "ttnn.experimental.">(
        mod,
        R"doc(
            Allocate the persistent stats DRAM scratch buffer required by
            `dit_fused_distributed_layernorm`'s all-gather path. LayerNorm transports 2
            stats/token (mean+var) vs RMS's 1, so this buffer is 2x wider than the RMS
            one — use this variant for LayerNorm ops. Returns None when the op needs no
            scratch (TP=1).
        )doc",
        &ttnn::experimental::dit_fused_distributed_layernorm_create_stats_buffer,
        nb::arg("input_tensor"),
        nb::arg("cluster_axis"),
        nb::arg("mesh_device"),
        nb::kw_only(),
        nb::arg("num_heads_per_device") = 1,
        nb::arg("num_links") = 1,
        nb::arg("weight") = nb::none(),
        nb::arg("transformation_mat") = nb::none(),
        nb::arg("rope_cos") = nb::none(),
        nb::arg("rope_sin") = nb::none());
}

}  // namespace ttnn::operations::experimental::ccl

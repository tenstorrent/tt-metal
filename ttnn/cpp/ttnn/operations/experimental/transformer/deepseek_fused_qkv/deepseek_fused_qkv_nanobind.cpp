// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "deepseek_fused_qkv_nanobind.hpp"

#include <cstdint>
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "deepseek_fused_qkv.hpp"

namespace ttnn::operations::experimental::transformer {

void bind_deepseek_fused_qkv(nb::module_& mod) {
    ttnn::bind_function<"deepseek_fused_qkv", "ttnn.experimental.">(
        mod,
        R"doc(
        Fused deepseek_v4_flash decode QKV projection (attention.py ``_qkv``).

        Fuses, in a single monolithic device op with DRAM-sharded weights:

          Q  : q_a = rmsnorm_w(hidden @ Wqa); q = q_a @ Wqb; reshape [1,1,H,Dh];
               per-head unweighted RMSNorm over Dh; partial RoPE on trailing rope_dim.
          KV : kv = rmsnorm_w(hidden @ Wkv); partial RoPE on trailing rope_dim.
               (runs concurrently on a disjoint parallel core partition)

        Args:
            hidden (ttnn.Tensor): decode activation ``[1, 1, 1, D]``.
            wqa (ttnn.Tensor): DRAM width-sharded q_a weight ``[D, q_lora]``.
            wqb (ttnn.Tensor): DRAM width-sharded q_b weight ``[q_lora, H*Dh]``.
            wkv (ttnn.Tensor): DRAM width-sharded kv weight ``[D, Dh]``.
            qa_norm_w (ttnn.Tensor): q_a RMSNorm gain ``[1, 1, 1, q_lora]``.
            kv_norm_w (ttnn.Tensor): kv RMSNorm gain ``[1, 1, 1, Dh]``.
            cos (ttnn.Tensor): DRAM-interleaved cos table ``[1, 1, 1, rope_dim]``.
            sin (ttnn.Tensor): DRAM-interleaved sin table ``[1, 1, 1, rope_dim]``.
            trans_mat (ttnn.Tensor): single ``[32, 32]`` rotate_half tile.
            eps (float): RMSNorm epsilon.
            rope_dim (int): trailing channel count that gets RoPE (tile-aligned).
            num_heads (int): number of query heads H.

        Keyword Args:
            q_mem_config (Optional[ttnn.MemoryConfig]): output memory config for q.
            kv_mem_config (Optional[ttnn.MemoryConfig]): output memory config for kv.
            compute_kernel_config (Optional[ttnn.DeviceComputeKernelConfig]): compute settings.

        Returns:
            List[ttnn.Tensor]: ``[q, kv]`` with q ``[1,1,H,Dh]`` and kv ``[1,1,1,Dh]``.
        )doc",
        &ttnn::experimental::deepseek_fused_qkv,
        nb::arg("hidden"),
        nb::arg("wqa"),
        nb::arg("wqb"),
        nb::arg("wkv"),
        nb::arg("qa_norm_w"),
        nb::arg("kv_norm_w"),
        nb::arg("cos"),
        nb::arg("sin"),
        nb::arg("trans_mat"),
        nb::arg("eps"),
        nb::arg("rope_dim"),
        nb::arg("num_heads"),
        nb::kw_only(),
        nb::arg("q_mem_config") = std::nullopt,
        nb::arg("kv_mem_config") = std::nullopt,
        nb::arg("compute_kernel_config") = std::nullopt);
}

}  // namespace ttnn::operations::experimental::transformer

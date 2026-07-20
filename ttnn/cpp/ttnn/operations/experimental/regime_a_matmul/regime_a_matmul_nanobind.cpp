// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "regime_a_matmul_nanobind.hpp"

#include <optional>

#include <fmt/format.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include "regime_a_matmul.hpp"
#include "device/regime_a_matmul_config.hpp"
#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/types.hpp"
#include "ttnn/device.hpp"
#include <tt-metalium/constants.hpp>

namespace ttnn::operations::experimental::regime_a_matmul::detail {

void bind_regime_a_matmul(nb::module_& mod) {
    ttnn::bind_function<"regime_a_matmul", "ttnn.experimental.">(
        mod,
        R"doc(
        regime_a_matmul(input_tensor, weight_tensor, config=None, *, bias_tensor=None, fused_activation=None, fused_ternary_scalar=None, fused_ternary_input_a=None, fused_ternary_input_b=None, memory_config=None, dtype=None, compute_kernel_config=None)

        Experimental DRAM-bandwidth-optimal matrix multiply (A @ B) for low-arithmetic-intensity
        (M << N or N << M) "Regime-A" shapes, with optional fused epilogue. bf16 in/out, HiFi2 math,
        fp32 dest accumulation.

        Fusions (applied at the output/compute stage; for split-K they run exactly once after reduction):
          - bias:       Y = A@B + bias
          - activation: Y = activation(A@B + bias)                (bias applied before activation)
          - addcmul:    Y = residual + scalar*(A@B + bias)*gate   (activation and addcmul are exclusive)

        The activation A ([.., M, K]) is DRAM interleaved. The weight B ([.., K, N]) must be DRAM
        WIDTH_SHARDED across 8 banks — build its MemoryConfig with
        ``ttnn.experimental.create_regime_a_weight_memory_config``. Output is [.., M, N] in TILE layout.

        Parameters
        ----------
        input_tensor : ttnn.Tensor
            Activation A. TILE layout, BFLOAT16, on device. Shape [.., M, K] (leading dims must be 1).
        weight_tensor : ttnn.Tensor
            Weight B. TILE layout, BFLOAT16, on device, DRAM WIDTH_SHARDED. Shape [.., K, N].
        config : Optional[RegimeAMatmulConfig], default: None
            Manual execution config. None => auto-select via the FLUX/LTX picker.
        bias_tensor : Optional[ttnn.Tensor], default: None
            Row-broadcast bias [.., 1, N] / [.., N], TILE, on device.
        fused_activation : Optional[UnaryWithParam], default: None
            Fused unary activation applied after bias.
        fused_ternary_scalar : Optional[float], default: None
            addcmul scalar. If set, fused_ternary_input_a (residual) and fused_ternary_input_b (gate)
            are required and fused_activation must be None.
        fused_ternary_input_a : Optional[ttnn.Tensor], default: None
            addcmul residual [M, N], BFLOAT16, TILE.
        fused_ternary_input_b : Optional[ttnn.Tensor], default: None
            addcmul gate [1, N] (broadcast) or [M, N] (full), TILE.
        memory_config : Optional[ttnn.MemoryConfig], default: None
            Output memory config. Defaults to DRAM interleaved.
        dtype : Optional[ttnn.DataType], default: None
            Output dtype. Defaults to BFLOAT16.
        compute_kernel_config : Optional[DeviceComputeKernelConfig], default: None
            Compute kernel config. Defaults to HiFi2 + fp32 accumulation.

        Returns
        -------
        ttnn.Tensor
            Output tensor [.., M, N], TILE layout.
        )doc",
        &ttnn::experimental::regime_a_matmul,
        nb::arg("input_tensor"),
        nb::arg("weight_tensor"),
        nb::arg("config") = nb::none(),
        nb::kw_only(),
        nb::arg("bias_tensor") = nb::none(),
        nb::arg("fused_activation") = nb::none(),
        nb::arg("fused_ternary_scalar") = nb::none(),
        nb::arg("fused_ternary_input_a") = nb::none(),
        nb::arg("fused_ternary_input_b") = nb::none(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("dtype") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none());

    ttnn::bind_function<"regime_a_matmul_split", "ttnn.experimental.">(
        mod,
        R"doc(
        regime_a_matmul_split(input_tensor, weight_tensor, chunks, dim=-1, config=None, *, bias_tensor=None, fused_activation=None, fused_ternary_scalar=None, fused_ternary_input_a=None, fused_ternary_input_b=None, memory_config=None, dtype=None, compute_kernel_config=None)

        Output column-split sibling of regime_a_matmul. Returns `chunks` equal-width [.., M, N/chunks]
        output tensors, written directly (no full-output materialize + slice). Requires dim==-1,
        N % chunks == 0 and N/chunks tile-aligned. All fusions compose with chunking.

        Returns
        -------
        List[ttnn.Tensor]
            `chunks` output tensors [.., M, N/chunks], TILE layout.
        )doc",
        &ttnn::experimental::regime_a_matmul_split,
        nb::arg("input_tensor"),
        nb::arg("weight_tensor"),
        nb::arg("chunks"),
        nb::arg("dim") = -1,
        nb::arg("config") = nb::none(),
        nb::kw_only(),
        nb::arg("bias_tensor") = nb::none(),
        nb::arg("fused_activation") = nb::none(),
        nb::arg("fused_ternary_scalar") = nb::none(),
        nb::arg("fused_ternary_input_a") = nb::none(),
        nb::arg("fused_ternary_input_b") = nb::none(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("dtype") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none());

    auto py_config = nb::class_<RegimeAMatmulConfig>(
                         mod,
                         "RegimeAMatmulConfig",
                         R"doc(
                         Configuration for the Regime-A matmul operation (all values in tiles / slice counts).
                         )doc")
                         .def(nb::init<>())
                         .def(
                             nb::init<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t>(),
                             nb::kw_only(),
                             nb::arg("k_slices") = 1,
                             nb::arg("n_slices") = 1,
                             nb::arg("m_slices") = 1,
                             nb::arg("k_block_tiles") = 1,
                             nb::arg("n_subblock_tiles") = 0);

    py_config.def_rw("k_slices", &RegimeAMatmulConfig::k_slices, "");
    py_config.def_rw("n_slices", &RegimeAMatmulConfig::n_slices, "");
    py_config.def_rw("m_slices", &RegimeAMatmulConfig::m_slices, "");
    py_config.def_rw("k_block_tiles", &RegimeAMatmulConfig::k_block_tiles, "");
    py_config.def_rw("n_subblock_tiles", &RegimeAMatmulConfig::n_subblock_tiles, "");
    // Build the repr manually (this file is compiled standalone / SKIP_UNITY, so the generic
    // reflection-based fmt formatter for aggregates is not in scope here).
    py_config.def("__repr__", [](const RegimeAMatmulConfig& c) {
        return "RegimeAMatmulConfig(k_slices=" + std::to_string(c.k_slices) +
               ", n_slices=" + std::to_string(c.n_slices) + ", m_slices=" + std::to_string(c.m_slices) +
               ", k_block_tiles=" + std::to_string(c.k_block_tiles) +
               ", n_subblock_tiles=" + std::to_string(c.n_subblock_tiles) + ")";
    });

    // Build the canonical DRAM width-sharded MemoryConfig for the in1 (weight) tensor.
    mod.def(
        "create_regime_a_weight_memory_config",
        [](const ttnn::Shape& weight_shape, tt::tt_metal::DataType dtype, ttnn::MeshDevice* device) {
            return ttnn::experimental::prim::create_regime_a_weight_memory_config(weight_shape, dtype, device);
        },
        nb::arg("weight_shape"),
        nb::arg("dtype"),
        nb::arg("device"),
        R"doc(
        create_regime_a_weight_memory_config(weight_shape, dtype, device)

        Return the DRAM WIDTH_SHARDED (8-bank, ROW_MAJOR) MemoryConfig required for the Regime-A matmul
        weight tensor. K is padded up to a multiple of 8 tiles, N up to a multiple of 8 tiles; the shard
        spec depends only on (K, N), never on the execution config.
        )doc");
}

}  // namespace ttnn::operations::experimental::regime_a_matmul::detail

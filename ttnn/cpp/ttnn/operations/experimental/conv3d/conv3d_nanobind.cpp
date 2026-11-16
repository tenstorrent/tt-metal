// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "conv3d_nanobind.hpp"

#include <array>
#include <cstdint>
#include <optional>

#include <fmt/format.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>

#include "conv3d.hpp"
#include "ttnn/operations/experimental/conv3d/prepare_conv3d_weights.hpp"
#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/types.hpp"
#include <tt-metalium/constants.hpp>

namespace ttnn::operations::experimental::conv3d::detail {

void bind_conv3d(nb::module_& mod) {
    ttnn::bind_function<"conv3d", "ttnn.experimental.">(
        mod,
        R"doc(
        Applies a 3D convolution over an input signal composed of several input planes. \
        Expects Input Tensor in [N, D, H, W, C] format.  \
        Expects Weight Tensor in [1, 1, kD * kH * kW * C_in, C_out] format. \
        Expects Bias Tensor in [1, 1, 1, 32, C_out] format. \
        Input must be in row major interleaved format. \
        Output will be in row major interleaved format.

        Args:
            input_tensor (ttnn.Tensor): Input tensor.
            weight_tensor (ttnn.Tensor): Weight tensor.
            config (ttnn.Conv3dConfig, optional): Configuration for the Conv3D operation. If not provided, conservative default blocking is used.

        Keyword Args:
            bias_tensor (ttnn.Tensor, optional): Bias tensor.
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the output of the Conv3D operation.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional): Compute kernel configuration for the Conv3D operation.
            weight_lo_tensor (ttnn.Tensor, optional): Required with ``config.enable_fp32_operand_split``: the residual ``W_lo = W - bf16(W)``. Split ``W`` into ``W_hi = bf16(W)`` and ``W_lo`` **before** preparation and pass both through ``prepare_conv3d_weights`` with identical ``groups``, ``alignment`` and ``C_in_block`` (or pass both raw, rank-5, and let this op prepare them). The kernel splits the fp32 activation into ``bf16(x)`` and its residual and accumulates ``x_hi*W_hi + x_hi*W_lo + x_lo*W_hi`` (dropping ``x_lo*W_lo``) in one fp32 pass; needs ``fp32_dest_acc_en``.

        Returns:
            ttnn.Tensor: Output tensor after applying the Conv3D operation.
        )doc",
        &ttnn::experimental::conv3d,
        nb::kw_only(),
        nb::arg("input_tensor"),
        nb::arg("weight_tensor"),
        nb::arg("device") = nb::none(),
        nb::arg("bias_tensor") = nb::none(),
        nb::arg("config") = nb::none(),
        nb::arg("dtype"),
        nb::arg("output_channels"),
        nb::arg("kernel_size"),
        nb::arg("stride") = std::array<uint32_t, 3>{1, 1, 1},
        nb::arg("padding") = std::array<uint32_t, 3>{0, 0, 0},
        nb::arg("dilation") = std::array<uint32_t, 3>{1, 1, 1},
        nb::arg("padding_mode") = "zeros",
        nb::arg("groups") = 1,
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none(),
        nb::arg("halo_buffer") = nb::none(),
        nb::arg("logical_h_mask") = 0u,
        nb::arg("logical_w_mask") = 0u,
        nb::arg("pad_offset_tensor") = nb::none(),
        nb::arg("output_pad_h") = 0u,
        nb::arg("output_pad_w") = 0u,
        nb::arg("weight_lo_tensor") = nb::none());

    // Register to ttnn.experimental namespace
    ttnn::bind_function<"prepare_conv3d_weights", "ttnn.experimental.">(
        mod,
        R"doc(Prepare conv3d weights for TTNN execution.

        For ``Conv3dConfig.enable_fp32_operand_split``, split the fp32 weight **before** preparing it:
        ``W_hi = W.to(bfloat16).to(float32)`` and ``W_lo = W - W_hi``, then prepare each with this function using
        identical ``groups``, ``C_in_block`` and ``alignment`` and pass them as ``weight_tensor`` /
        ``weight_lo_tensor``. Splitting after preparation, or preparing the two with different arguments, silently
        pairs mismatched rows.)doc",
        &ttnn::operations::experimental::conv3d::prepare_conv3d_weights,
        nb::kw_only(),
        nb::arg("weight_tensor"),
        nb::arg("groups") = 1u,
        // 0 == use the default minimal valid block (default_c_in_block), the same value conv3d
        // defaults to, so the prepared weight and the conv compute agree on K-row blocking and
        // stay within L1. A mismatch silently reorders rows (issues #42146, #47316).
        nb::arg("C_in_block") = 0u,
        nb::arg("alignment") = 32u,
        nb::arg("device") = nb::none());

    auto py_conv3d_config = nb::class_<ttnn::experimental::prim::Conv3dConfig>(
                                mod,
                                "Conv3dConfig",
                                R"doc(
                            Configuration for the Conv3D operation.
                            )doc")
                                .def(nb::init<>())
                                .def(
                                    nb::init<
                                        DataType,
                                        Layout,
                                        uint32_t,
                                        uint32_t,
                                        uint32_t,
                                        uint32_t,
                                        uint32_t,
                                        std::array<uint32_t, 3>,
                                        uint32_t,
                                        CoreCoord,
                                        bool>(),
                                    nb::kw_only(),
                                    nb::arg("weights_dtype") = DataType::BFLOAT16,
                                    nb::arg("output_layout") = Layout::ROW_MAJOR,
                                    nb::arg("T_out_block") = 1,
                                    nb::arg("W_out_block") = 1,
                                    nb::arg("H_out_block") = 1,
                                    nb::arg("C_out_block") = 0,
                                    nb::arg("C_in_block") = 0,
                                    nb::arg("dilation") = std::array<uint32_t, 3>{1, 1, 1},
                                    nb::arg("alignment") = 32,
                                    nb::arg("compute_with_storage_grid_size") = nb::cast(CoreCoord{1, 1}),
                                    nb::arg("enable_fp32_operand_split") = false);

    py_conv3d_config.def_rw("weights_dtype", &ttnn::experimental::prim::Conv3dConfig::weights_dtype, "");
    py_conv3d_config.def_rw("output_layout", &ttnn::experimental::prim::Conv3dConfig::output_layout, "");
    py_conv3d_config.def_rw("T_out_block", &ttnn::experimental::prim::Conv3dConfig::T_out_block, "");
    py_conv3d_config.def_rw("W_out_block", &ttnn::experimental::prim::Conv3dConfig::W_out_block, "");
    py_conv3d_config.def_rw("H_out_block", &ttnn::experimental::prim::Conv3dConfig::H_out_block, "");
    py_conv3d_config.def_rw("C_out_block", &ttnn::experimental::prim::Conv3dConfig::C_out_block, "");
    py_conv3d_config.def_rw("alignment", &ttnn::experimental::prim::Conv3dConfig::alignment, "");
    py_conv3d_config.def_rw("C_in_block", &ttnn::experimental::prim::Conv3dConfig::C_in_block, "");
    py_conv3d_config.def_rw("dilation", &ttnn::experimental::prim::Conv3dConfig::dilation, "");
    py_conv3d_config.def_rw(
        "compute_with_storage_grid_size", &ttnn::experimental::prim::Conv3dConfig::compute_with_storage_grid_size, "");
    py_conv3d_config.def_rw(
        "enable_fp32_operand_split",
        &ttnn::experimental::prim::Conv3dConfig::enable_fp32_operand_split,
        "FP32-only precision mode: splits the fp32 activation into bf16(x) and its residual and accumulates "
        "x_hi*W_hi + x_hi*W_lo + x_lo*W_hi in one pass (x_lo*W_lo is dropped). Requires float32 input, "
        "``weight_lo_tensor`` (the residual W - bf16(W), prepared with the same prepare_conv3d_weights arguments as "
        "the weight) and ``compute_kernel_config.fp32_dest_acc_en``. Costs three matmul products per block.");

    py_conv3d_config.def(
        "__repr__", [](const ttnn::experimental::prim::Conv3dConfig& config) { return fmt::format("{}", config); });
}

}  // namespace ttnn::operations::experimental::conv3d::detail

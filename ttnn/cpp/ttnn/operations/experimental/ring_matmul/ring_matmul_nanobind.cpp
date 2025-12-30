// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ring_matmul_nanobind.hpp"

#include <optional>

#include <fmt/format.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ring_matmul.hpp"
#include "ttnn-nanobind/decorators.hpp"
#include "ttnn/types.hpp"
#include <tt-metalium/constants.hpp>

namespace ttnn::operations::experimental::ring_matmul::detail {

void bind_ring_matmul(nb::module_& mod) {
    bind_registered_operation(
        mod,
        ttnn::experimental::ring_matmul,
        R"doc(
        ring_matmul(input_tensor, weight_tensor, *, fused_activation=None, config=None, memory_config=None, dtype=None, compute_kernel_config=None, hop_cores=CoreRangeSet{}, global_cb=None, num_global_cb_receivers=1, sub_device_id=None, restricted_cores=None, untilize_out=False)

        Experimental ring-based matrix multiply operation with all-gather pattern.
        This op performs matmul with a ring all-gather pattern on the input tensor, where shards are
        circulated around a ring of cores for computation.

        Parameters
        ----------
        input_tensor : ttnn.Tensor
            Activation/input matrix A.
            - Layout: TILE (required).
            - Device: must be on device and allocated in a device buffer.
            - Must be sharded (width sharded along K dimension).

        weight_tensor : ttnn.Tensor
            Weight matrix B.
            - Layout: TILE (required).
            - Device: same device as `input_tensor`.
            - Shape: [..., K, N].

        Returns
        -------
        ttnn.Tensor
            Output tensor with shape [..., M, N], TILE layout.
        )doc",
        ttnn::nanobind_arguments_t{
            nb::arg("input_tensor"),
            nb::arg("weight_tensor"),
            nb::kw_only(),
            nb::arg("fused_activation") = nb::none(),
            nb::arg("config") = nb::none(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("dtype") = nb::none(),
            nb::arg("compute_kernel_config") = nb::none(),
            nb::arg("hop_cores") = CoreRangeSet{},
            nb::arg("global_cb") = nb::none(),
            nb::arg("num_global_cb_receivers") = 1,
            nb::arg("sub_device_id") = nb::none(),
            nb::arg("restricted_cores") = nb::none(),
            nb::arg("untilize_out") = false});

    auto py_ring_matmul_config =
        nb::class_<RingMatmulConfig>(mod, "RingMatmulConfig", R"doc(Configuration for the RingMatmul operation.)doc")
            .def(nb::init<>())
            .def(
                nb::init<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, bool, bool, bool>(),
                nb::kw_only(),
                nb::arg("in0_block_w") = 1,
                nb::arg("out_subblock_h") = 1,
                nb::arg("out_subblock_w") = 1,
                nb::arg("per_core_M") = 1,
                nb::arg("per_core_N") = 1,
                nb::arg("packer_l1_acc") = true,
                nb::arg("fp32_dest_acc_en") = false,
                nb::arg("dst_full_sync_en") = false);

    py_ring_matmul_config.def_rw("in0_block_w", &RingMatmulConfig::in0_block_w, "");
    py_ring_matmul_config.def_rw("out_subblock_h", &RingMatmulConfig::out_subblock_h, "");
    py_ring_matmul_config.def_rw("out_subblock_w", &RingMatmulConfig::out_subblock_w, "");
    py_ring_matmul_config.def_rw("per_core_M", &RingMatmulConfig::per_core_M, "");
    py_ring_matmul_config.def_rw("per_core_N", &RingMatmulConfig::per_core_N, "");
    py_ring_matmul_config.def_rw("packer_l1_acc", &RingMatmulConfig::packer_l1_acc, "");
    py_ring_matmul_config.def_rw("fp32_dest_acc_en", &RingMatmulConfig::fp32_dest_acc_en, "");
    py_ring_matmul_config.def_rw("dst_full_sync_en", &RingMatmulConfig::dst_full_sync_en, "");

    py_ring_matmul_config.def("__repr__", [](const RingMatmulConfig& config) { return fmt::format("{}", config); });
}

}  // namespace ttnn::operations::experimental::ring_matmul::detail

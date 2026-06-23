// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "slice_nanobind.hpp"

#include <array>
#include <cstdint>
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/small_vector_caster.hpp"
#include "ttnn-nanobind/bind_function.hpp"

#include "slice.hpp"
#include "ttnn/operations/experimental/quasar/slice/device/slice_device_operation.hpp"

namespace ttnn::operations::experimental::quasar::detail {

namespace {

ttnn::Tensor slice_small_vector_wrapper(
    const ttnn::Tensor& input_tensor,
    const ttnn::SmallVector<int>& slice_start,
    const ttnn::SmallVector<int>& slice_end,
    const std::optional<ttnn::SmallVector<int>>& step,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor,
    const std::optional<float>& pad_value,
    const std::optional<CoreRangeSet>&& sub_core_grids) {
    const auto step_value = step.value_or(ttnn::SmallVector<int>(slice_end.size(), 1));
    return ttnn::operations::experimental::quasar::slice(
        input_tensor,
        slice_start,
        slice_end,
        step_value,
        memory_config,
        optional_output_tensor,
        pad_value,
        sub_core_grids);
}

}  // namespace

void bind_slice(nb::module_& mod) {
    const auto* doc = R"doc(
        Returns a sliced tensor. If the input tensor is on host, the slice will be performed on host, and if its on device it will be performed on device.

        Args:
            input_tensor (ttnn.Tensor): Input tensor.
            slice_start (List[int]): Start indices of input tensor. Values along each dim must be in ``[0, input_tensor_shape[i])``.
            slice_end (List[int]): End indices of input tensor (exclusive). Values along each dim must be in ``(0, input_tensor_shape[i]]``.
            slice_step (List[int], optional): Step size for each dim. Defaults to ``None`` (step = 1 for all dims).

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the output tensor. Defaults to the input tensor's memory config.
            output_tensor (ttnn.Tensor, optional): Pre-allocated output tensor. Its shape must match the slice output. Defaults to ``None``.
            pad_value (float, optional): Fill value for implicit tile padding on tiled tensors. Padding is undefined by default.
            sub_core_grids (ttnn.CoreRangeSet, optional): sub core grids for the operation. Defaults to `None`.

        Note:
            Strided slicing (``slice_step != 1``) is not supported for ``bfloat8_b`` tensors.

        Returns:
            ttnn.Tensor: the output tensor.
    )doc";

    // TODO: implementing the array version and overloading the nanobind with all the possible array sizes is better
    // than a vector with a fixed size default value
    ttnn::bind_function<"slice", "ttnn.experimental.quasar.">(
        mod,
        doc,
        // Overload 1: Tensor args version (uint32_t template parameter)
        ttnn::overload_t(
            nb::overload_cast<
                const ttnn::Tensor&,
                const ttnn::Tensor&,
                const ttnn::Tensor&,
                const std::optional<ttnn::SmallVector<uint32_t>>&,
                const std::optional<MemoryConfig>&,
                const std::optional<Tensor>&,
                const std::optional<float>&,
                const std::optional<uint32_t>&,
                const std::optional<uint32_t>&,
                const std::optional<CoreRangeSet>&>(&ttnn::operations::experimental::quasar::slice<uint32_t>),
            nb::arg("input_tensor"),
            nb::arg("starts"),
            nb::arg("ends"),
            nb::arg("slice_step") = nb::none(),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none(),
            nb::arg("pad_value") = nb::none(),
            nb::arg("slice_dim") = nb::none(),
            nb::arg("num_devices") = nb::none(),
            nb::arg("sub_core_grids") = nb::none()),
        // Overload 2: std::array version (uint32_t template parameter, size 4)
        ttnn::overload_t(
            nb::overload_cast<
                const ttnn::Tensor&,
                const std::array<uint32_t, 4>&,
                const std::array<uint32_t, 4>&,
                const std::array<uint32_t, 4>&,
                const std::optional<MemoryConfig>&,
                const std::optional<Tensor>&,
                const std::optional<float>&,
                const std::optional<CoreRangeSet>&>(&ttnn::operations::experimental::quasar::slice<uint32_t, 4>),
            nb::arg("input_tensor"),
            nb::arg("starts"),
            nb::arg("ends"),
            nb::arg("steps"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none(),
            nb::arg("pad_value") = nb::none(),
            nb::arg("sub_core_grids") = nb::none()),
        // Overload 3: SmallVector<int> version (int32_t template parameter)
        ttnn::overload_t(
            &slice_small_vector_wrapper,
            nb::arg("input_tensor"),
            nb::arg("slice_start"),
            nb::arg("slice_end"),
            nb::arg("slice_step") = nb::none(),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none(),
            nb::arg("pad_value") = nb::none(),
            nb::arg("sub_core_grids") = nb::none()));
}
void bind_slice_descriptor(nb::module_& mod) {
    nb::class_<ttnn::prim::qsr::SliceParams>(mod, "SliceParams")
        .def(nb::init<>())
        .def_rw("slice_start", &ttnn::prim::qsr::SliceParams::slice_start)
        .def_rw("slice_end", &ttnn::prim::qsr::SliceParams::slice_end)
        .def_rw("step", &ttnn::prim::qsr::SliceParams::step)
        .def_rw("output_mem_config", &ttnn::prim::qsr::SliceParams::output_mem_config)
        .def_rw("sub_core_grids", &ttnn::prim::qsr::SliceParams::sub_core_grids);

    nb::class_<ttnn::prim::qsr::SliceInputs>(mod, "SliceInputs")
        .def(
            "__init__",
            [](ttnn::prim::qsr::SliceInputs* t, const ttnn::Tensor& input) {
                new (t) ttnn::prim::qsr::SliceInputs{input, std::nullopt, std::nullopt, std::nullopt};
            },
            nb::arg("input"))
        .def_rw("input", &ttnn::prim::qsr::SliceInputs::input)
        .def_rw("preallocated_output", &ttnn::prim::qsr::SliceInputs::preallocated_output);

    nb::class_<ttnn::prim::qsr::SliceDeviceOperation>(mod, "SliceDeviceOperation")
        .def_static(
            "create_output_tensors",
            &ttnn::prim::qsr::SliceDeviceOperation::create_output_tensors,
            nb::arg("operation_attributes"),
            nb::arg("tensor_args"))
        .def_static(
            "compute_output_specs",
            &ttnn::prim::qsr::SliceDeviceOperation::compute_output_specs,
            nb::arg("operation_attributes"),
            nb::arg("tensor_args"));
}

}  // namespace ttnn::operations::experimental::quasar::detail
